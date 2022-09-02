import math
from functools import partial

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.base import ALL
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init
from torch.utils.checkpoint import checkpoint


class GATConv(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        norm='none',
    ):
        super(GATConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm

        # feat fc
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
        else:
            self.attn_edge_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["sub_gcn_norm_adjust"][eids].view(-1, 1, 1) 
            if self._norm == "avg":
                graph.edata["a"][eids] = (graph.edata["a"][eids] + graph.edata["sub_gcn_norm"][eids].view(-1, 1, 1)) / 2

            # message passing
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
            rst = graph.dstdata["feat_src_fc"]

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst


class GAT(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        edge_emb,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        residual=False,
        norm='none',
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else node_feats
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if self.edge_encoder is not None:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
            self.convs.append(
                GATConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,  # TAG
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.residual = residual

    def forward(self, g, inference=False):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if self.residual and h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)

        return h

class AGDNConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        n_heads=1,
        K=3,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        norm="none",
        batch_norm=True,
    ):
        super(AGDNConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        self._K = K
        self._batch_norm = batch_norm

        # feat fc
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(torch.FloatTensor(out_feats * n_heads))

        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(K + 1):
                self.offset.append(nn.Parameter(torch.zeros(size=(1, n_heads, out_feats))))
                self.scale.append(nn.Parameter(torch.ones(size=(1, n_heads, out_feats))))

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation
        self.hop_attn_l = nn.Parameter(torch.Tensor(size=(1, n_heads, out_feats)))
        self.hop_attn_r = nn.Parameter(torch.Tensor(size=(1, n_heads, out_feats)))
        self.position_emb = nn.Parameter(torch.Tensor(K, n_heads, out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)

        nn.init.zeros_(self.hop_attn_l)
        nn.init.zeros_(self.hop_attn_r)
        for emb in self.position_emb:
            nn.init.xavier_normal_(emb, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def feat_trans(self, h, idx):
        
        if self._batch_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]
        h = h + self.position_emb[[idx], :, :]
        return h

    def forward(self, graph, feat_src):
        with graph.local_scope():

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src
            

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]

            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)
            graph.edata["a"][eids] = self.attn_drop(
                                        torch.sqrt(edge_softmax(graph, e[eids], eids=eids, norm_by='dst').clamp(min=1e-9) \
                                                 * edge_softmax(graph, e[eids], eids=eids, norm_by='src').clamp(min=1e-9)))
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["sub_gcn_norm_adjust"][eids].view(-1, 1, 1) 
            if self._norm == "avg":
                graph.edata["a"][eids] = (graph.edata["a"][eids] + graph.edata["sub_gcn_norm"][eids].view(-1, 1, 1)) / 2

            # message passing
            hstack = []
            for k in range(self._K):
                graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))

                hstack.append(graph.dstdata["feat_src_fc"])

            hstack = [self.feat_trans(h, k) for k, h in enumerate(hstack)]
            a_l = (hstack[0] * self.hop_attn_l).sum(-1).unsqueeze(-1)
            astack_r = [(hstack[k] * self.hop_attn_r).sum(-1).unsqueeze(-1) for k in range(len(hstack))]
            a = torch.cat([a_r + a_l for a_r in astack_r], dim=-1)
            a = F.softmax(self.leaky_relu(a), dim=-1)
            rst = 0
            for k in range(self._K):
                rst += hstack[k] * a[:, :, [k]]


            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst

class AGDN(nn.Module):
    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        K=3,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        norm="none",
        residual=False,
        shadow=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.residual = residual
        self.shadow = shadow

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else node_feats
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            self.convs.append(
                AGDNConv(
                    in_hidden,
                    out_hidden,
                    n_heads=n_heads,
                    K=K,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        if self.shadow:
            self.pred_linear = nn.Linear(2 * n_heads * n_hidden, n_classes)
        else:
            self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]

        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):

            h = self.convs[i](subgraphs[i], h).flatten(1, -1)

            if h_last is not None and self.residual:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)
        if self.shadow:
            h = torch.cat([h, h.mean(dim=0, keepdim=True)], dim=1)
        h = self.pred_linear(h)
        return h
