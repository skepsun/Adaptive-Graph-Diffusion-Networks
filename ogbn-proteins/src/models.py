from dgl.batch import batch
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn.modules.dropout import Dropout


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
        norm="none",
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
            self.dst_fc = self.src_fc
            self.bias = nn.Parameter(torch.FloatTensor(out_feats * n_heads))

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
                
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)
            graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
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
        allow_zero_in_degree=False,
        norm="none",
        use_one_hot_feature=False,
        use_labels=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if use_one_hot_feature:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if edge_emb > 0:
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
                    norm=norm,
                )
            )
            self.norms.append(nn.BatchNorm1d(out_hidden * n_heads))

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
        x = subgraphs[0].srcdata["x"]
        

        if self.one_hot_encoder is not None:
            x = self.one_hot_encoder(x)
            h = torch.cat([x, h], dim=1)

        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
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

            if h_last is not None:
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
        edge_feats,
        out_feats,
        n_heads=1,
        K=3,
        attn_drop=0.0,
        hop_attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        norm="none",
        batch_norm=True,
        weight_style="HA",
    ):
        super(AGDNConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        self._batch_norm = batch_norm
        self._K = K
        self._weight_style = weight_style

        # feat fc
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = self.src_fc
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_feats)))
    
        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
            self.edge_norm = nn.BatchNorm1d(edge_feats)
        else:
            self.attn_edge_fc = None
            self.edge_norm = None
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(K + 1):
                self.offset.append(nn.Parameter(torch.zeros(size=(1, n_heads, out_feats))))
                self.scale.append(nn.Parameter(torch.ones(size=(1, n_heads, out_feats))))
        self.attn_drop = nn.Dropout(attn_drop)
        self.hop_attn_drop = hop_attn_drop
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation
        self.position_emb = nn.Parameter(torch.Tensor(K+1, n_heads, out_feats))
        if weight_style == "HA":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(n_heads, out_feats)))
            self.hop_attn_l_bias = nn.Parameter(torch.FloatTensor(size=(n_heads, out_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(n_heads, out_feats)))
            self.hop_attn_r_bias = nn.Parameter(torch.FloatTensor(size=(n_heads, out_feats)))
        
        if weight_style == "HC":
            self.weights = nn.Parameter(torch.FloatTensor(size=(1, n_heads, K, out_feats)))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        # nn.init.zeros_(self.attn_src_fc.bias)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_dst_fc.bias)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_edge_fc.bias)

        if self._weight_style == "HA":
            nn.init.zeros_(self.hop_attn_l)
            nn.init.zeros_(self.hop_attn_l_bias)
            nn.init.zeros_(self.hop_attn_r)
            nn.init.zeros_(self.hop_attn_r_bias)
        for emb in self.position_emb:
            nn.init.xavier_normal_(emb, gain=gain)
        if self._weight_style == "HC":
            nn.init.ones_(self.weights)
            # nn.init.xavier_uniform_(self.weights, gain=gain)

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

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            if self.dst_fc is not None:
                feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
            else:
                feat_dst_fc = feat_src_fc
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
                
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)
            # graph.edata["a"][eids] = self.attn_drop((edge_softmax(graph, e[eids], eids=eids, norm_by='dst')))
            graph.edata["a"][eids] = self.attn_drop(
                                        torch.sqrt(edge_softmax(graph, e[eids], eids=eids, norm_by='dst').clamp(min=1e-9) \
                                                 * edge_softmax(graph, e[eids], eids=eids, norm_by='src').clamp(min=1e-9)))
            # graph.edata["a"][eids] = self.attn_drop(e[eids])
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm_adjust"][eids].view(-1, 1, 1) 
            if self._norm == "avg":
                graph.edata["a"][eids] = (graph.edata["a"][eids] + graph.edata["gcn_norm"][eids].view(-1, 1, 1)) / 2

            # message passing
            h_0 = self.feat_trans(graph.dstdata["feat_src_fc"], 0)
            hstack = []
            for k in range(self._K):
                graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
                # graph.dstdata["feat_src_fc"] = graph.dstdata["feat_src_fc"] / graph.ndata["sub_deg"].view(-1, 1, 1)
                hstack.append(graph.dstdata["feat_src_fc"])

            hstack = torch.stack([self.feat_trans(h, k+1) for k, h in enumerate(hstack)], dim=2)
            if self._weight_style == "sum":
                rst = hstack.sum(2)
            if self._weight_style == "mean":
                rst = hstack.mean(2)
            if self._weight_style == "HC":
                if self.training:
                    mask = torch.rand_like(self.weights) > self.hop_attn_drop
                else:
                    mask = torch.ones_like(self.weights).bool()
                weights = torch.ones_like(self.weights, device=self.weights.device)
                weights[mask] = self.weights[mask]
                rst = (hstack * weights).sum(2)
            if self._weight_style == "HA":
                a_l = (h_0.unsqueeze(2) * self.hop_attn_l.unsqueeze(0).unsqueeze(2)).sum(dim=-1, keepdim=True)
                a = (hstack * self.hop_attn_r.unsqueeze(0).unsqueeze(2)).sum(dim=-1, keepdim=True)
                a = a + a_l
                # a = torch.sigmoid(a)
                a = self.hop_attn_drop(a)
                a = F.softmax(self.leaky_relu(a), dim=-2)
                a = a.transpose(-2, -1)
                rst = torch.matmul(a, hstack).squeeze(-2)
           
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
        hop_attn_drop,
        edge_drop,
        K=3,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        norm="none",
        use_one_hot=False,
        use_labels=False,
        edge_attention=False,
        weight_style="HA",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        if edge_attention:
            self.pre_aggregator = EdgeAttentionLayer(edge_feats, n_heads)
        else:
            self.pre_aggregator = None
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                AGDNConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    K=K,
                    attn_drop=attn_drop,
                    hop_attn_drop=hop_attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    residual=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    weight_style=weight_style,
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g
        if self.pre_aggregator is not None:
            h = self.pre_aggregator(subgraphs[0])
        else:
            h = subgraphs[0].srcdata["feat"]

        if self.one_hot_encoder is not None:
            # x = F.relu(self.one_hot_encoder(x))
            x = subgraphs[0].srcdata["x"]
            h = torch.cat([x, h], dim=1)
        
        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):

            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                # efeat_emb = self.edge_norms[0](efeat_emb)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)
        # if self.label_encoder is not None:
        #     h += self.label_encoder(l)
        return h

class EdgeAttentionLayer(nn.Module):
    def __init__(self, e_feats, n_heads, edge_drop=0.0):
        super().__init__()
        self._n_heads = n_heads
        self._e_feats = e_feats
        self.edge_drop = edge_drop
        # self.fc = nn.Linear(e_feats, e_feats * n_heads, bias=False)
        self.att = nn.Parameter(torch.FloatTensor(size=(1, n_heads, e_feats)))
        # self.att_bias = nn.Parameter(torch.FloatTensor(size=(1, n_heads, e_feats)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.zeros_(self.att)
        # nn.init.zeros_(self.att_bias)
    
    def forward(self, graph):
        edge_feats = graph.edata["feat"].unsqueeze(1)
        # h_e = self.fc(edge_feats).view(-1, self._n_heads, self._e_feats)
        e = (edge_feats * self.att).sum(dim=-1, keepdim=True)
        # if self.training and self.edge_drop > 0:
        #     perm = torch.randperm(graph.number_of_edges(), device=e.device)
        #     bound = int(graph.number_of_edges() * self.edge_drop)
        #     eids = perm[bound:]
        # else:
        eids = torch.arange(graph.number_of_edges(), device=e.device)
        graph.edata["a"] = torch.zeros_like(e)
        graph.edata["a"][eids] = edge_softmax(graph, e[eids])
        graph.edata["f"] = edge_feats * graph.edata.pop("a")
        graph.update_all(fn.copy_e("f", "m"), fn.sum("m", "f"))
        h = graph.ndata.pop("f")
        h = h.mean(1)
        return h