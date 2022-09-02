import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn.modules.linear import Linear

# implementation from @Espylapiza
class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

# class GCN(nn.Module):
#     def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, residual):
#         super().__init__()
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.residual = residual

#         self.convs = nn.ModuleList()
#         if residual:
#             self.linear = nn.ModuleList()
#         self.bns = nn.ModuleList()

#         for i in range(n_layers):
#             in_hidden = n_hidden if i > 0 else in_feats
#             out_hidden = n_hidden if i < n_layers - 1 else n_classes
#             bias = i == n_layers - 1

#             self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
#             if residual:
#                 self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
#             if i < n_layers - 1:
#                 self.bns.append(nn.BatchNorm1d(out_hidden))

#         self.dropout0 = nn.Dropout(min(0.1, dropout))
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation

#     def forward(self, graph, feat):
#         h = feat
#         h = self.dropout0(h)

#         for i in range(self.n_layers):
#             conv = self.convs[i](graph, h)

#             if self.use_linear:
#                 linear = self.linear[i](h)
#                 h = conv + linear
#             else:
#                 h = conv

#             if i < self.n_layers - 1:
#                 h = self.bns[i](h)
#                 h = self.activation(h)
#                 h = self.dropout(h)

#         return h

'''
We conclude GCN and GAT into a general MPNN module. 
To identify them, use the argument 'transition matrix'. 
Note that here we do not use graph diffusion.
'''

class MPNNConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        transition_matrix="gat",
        bias=True,
    ):
        super(MPNNConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._transition_matrix = transition_matrix

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        if transition_matrix.startswith("gat"):
            self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            if use_attn_dst:
                self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            else:
                self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))
        else:
            self.bias = None
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self._transition_matrix.startswith("gat"):
            nn.init.xavier_normal_(self.attn_l, gain=gain)
            if isinstance(self.attn_r, nn.Parameter):
                nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src


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
            

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=graph.device)

            graph.srcdata.update({"ft": feat_src})
            if self._transition_matrix.startswith("gat"):
                el = (feat_src * self.attn_l).sum(-1).unsqueeze(-1)
                graph.srcdata.update({"el": el})
                # graph.dstdata.update({"er": er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                if self.attn_r is not None:
                    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.dstdata.update({"er": er})
                    graph.apply_edges(fn.u_add_v("el", "er", "e"))
                else:
                    graph.apply_edges(fn.copy_u("el", "e"))
                e = self.leaky_relu(graph.edata.pop("e"))
                
                # compute softmax
                
                
                a = edge_softmax(graph, e[eids], eids=eids)

                if self._transition_matrix == "gat_adj":
                    a = a * graph.edata["gcn_norm_adjust"][eids].unsqueeze(1).unsqueeze(1)
                if self._transition_matrix == "gat_sym":
                    a = torch.sqrt(a.clamp(min=1e-9) * edge_softmax(graph, e[eids], eids=eids, norm_by='src').clamp(min=1e-9))
            elif self._transition_matrix == "gcn":
                a = graph.edata["gcn_norm"][eids].unsqueeze(1).unsqueeze(1)
            elif self._transition_matrix == "sage":
                a = graph.edata["sage_norm"][eids].unsqueeze(1).unsqueeze(1)

            graph.edata["a"] = torch.zeros(size=(graph.number_of_edges(), self._num_heads, 1), device=feat_src.device)
            graph.edata["a"][eids] = self.attn_drop(a)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class MPNN(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        transition_matrix="gat",
        residual=True,
        bias_last=True,
        no_bias=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                MPNNConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    transition_matrix=transition_matrix,
                    residual=residual,
                    bias=(not bias_last) and (not no_bias),
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        if bias_last and (not no_bias):
            self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        else:
            self.bias_last = None

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        if self.bias_last is not None:
            h = self.bias_last(h)

        return h

'''
Our proposed AGDN actually contains AGDN_GCN, AGDN_GAT, AGDN_GAT_sym and AGDN_GAT_adj, 
which depends on the transition matrix.
'''

class AGDNConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        K=3,
        num_heads=1,
        feat_drop=0.0,
        edge_drop=0.0,
        attn_drop=0.0,
        diffusion_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        transition_matrix='gat_adj',
        weight_style="HA",
        HA_activation="leakyrelu",
        position_emb=True,
        batch_norm=False,
        propagate_first=False,
        zero_inits=False,
        bias=True,
    ):
        super(AGDNConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._K = K
        self._transition_matrix = transition_matrix
        self._weight_style = weight_style
        self._HA_activation = HA_activation
        self._position_emb = position_emb
        self._batch_norm = batch_norm
        self._propagate_first = propagate_first
        self._zero_inits = zero_inits

        if propagate_first:
            propagate_feats = in_feats
        else:
            propagate_feats = out_feats
        if transition_matrix.startswith('gat'):
            self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
            if use_attn_dst:
                if transition_matrix == 'gat_sym':
                    self.attn_r = self.attn_l
                else:
                    self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
            else:
                self.register_buffer("attn_r", None)

        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        if position_emb:
            self.position_emb = nn.Parameter(torch.FloatTensor(size=(K+1, num_heads, propagate_feats)))
        if weight_style in ["HA", "HA+HC"]:
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
            # self.hop_attn_bias_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))
            # self.hop_attn_bias_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))
            # self.beta = nn.Parameter(torch.FloatTensor(size=(num_heads,)))
        if weight_style in ["HC", "HA+HC"]:
            self.weights = nn.Parameter(torch.FloatTensor(size=(1, num_heads, K+1, propagate_feats)))

        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(K + 1):
                self.offset.append(nn.Parameter(torch.zeros(size=(1, num_heads, out_feats))))
                self.scale.append(nn.Parameter(torch.ones(size=(1, num_heads, out_feats))))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.diffusion_drop = diffusion_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self._transition_matrix.startswith('gat'):
            nn.init.xavier_normal_(self.attn_l, gain=gain)
            if isinstance(self.attn_r, nn.Parameter):
                nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._position_emb:
            nn.init.xavier_normal_(self.position_emb)
        if self._weight_style in ["HA", "HA+HC"]:
            if self._zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
            # nn.init.xavier_normal_(self.hop_attn_bias_l, gain=gain)
            # nn.init.xavier_normal_(self.hop_attn_bias_r, gain=gain)
            # nn.init.uniform_(self.beta)
        if self._weight_style in ["HC", "HA+HC"]:
            nn.init.xavier_uniform_(self.weights, gain=gain)
            # nn.init.ones_(self.weights)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def feat_trans(self, h, idx):
        
        if self._batch_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._num_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._num_heads, 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]

        if self._position_emb:
            h = h + self.position_emb[[idx], :, :]
        return h

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                if not self._propagate_first:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = feat_src.view(-1, 1, self._in_src_feats)
                    feat_dst = feat_dst.view(-1, 1, self._in_dst_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                if not self._propagate_first:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = feat_src.view(-1, 1, self._in_src_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src
            
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=graph.device)

            graph.srcdata.update({"ft": feat_src})
            if self._transition_matrix.startswith("gat"):
                el = (feat_src * self.attn_l).sum(-1).unsqueeze(-1)
                graph.srcdata.update({"el": el})
                # graph.dstdata.update({"er": er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                if self.attn_r is not None:
                    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.dstdata.update({"er": er})
                    graph.apply_edges(fn.u_add_v("el", "er", "e"))
                else:
                    graph.apply_edges(fn.copy_u("el", "e"))
                e = self.leaky_relu(graph.edata.pop("e"))
                
                # compute softmax
                
                
                a = edge_softmax(graph, e[eids], eids=eids)

                if self._transition_matrix == "gat_adj":
                    a = a * graph.edata["gcn_norm_adjust"][eids].unsqueeze(1).unsqueeze(1)
                if self._transition_matrix == "gat_sym":
                    a = torch.sqrt(a.clamp(min=1e-9) * edge_softmax(graph, e[eids], eids=eids, norm_by='src').clamp(min=1e-9))
            elif self._transition_matrix == "gcn":
                a = graph.edata["gcn_norm"][eids].unsqueeze(1).unsqueeze(1)
            elif self._transition_matrix == "sage":
                a = graph.edata["sage_norm"][eids].unsqueeze(1).unsqueeze(1)
            
            graph.edata["a"] = torch.zeros(size=(graph.number_of_edges(), self._num_heads, 1), device=feat_src.device)
            graph.edata["a"][eids] = self.attn_drop(a)
            
            hstack = [graph.dstdata["ft"]]

            for _ in range(self._K):
                # message passing
                if self.diffusion_drop > 0:
                    # We could choose to simulate the dropout between convolutions by setting diffusion_drop > 0
                    graph.ndata['ft'] = F.dropout(graph.ndata['ft'], self.diffusion_drop, training=self.training)
                graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

                hstack.append(graph.dstdata["ft"])

            hstack = [self.feat_trans(h, k) for k, h in enumerate(hstack)]

            hop_a = None
            if self._weight_style in ["HA", "HA+HC"]:
                hop_a_l = (hstack[0] * self.hop_attn_l).sum(dim=-1).unsqueeze(-1)
                hop_astack_r = [(feat_dst * self.hop_attn_r).sum(dim=-1).unsqueeze(-1) for feat_dst in hstack]
                hop_a = torch.cat([(a_r + hop_a_l) for a_r in hop_astack_r], dim=-1)
                if self._HA_activation == "sigmoid":
                    hop_a = torch.sigmoid(hop_a)
                if self._HA_activation == "leakyrelu":
                    hop_a = self.leaky_relu(hop_a)
                if self._HA_activation == "relu":
                    hop_a = F.relu(hop_a)
                if self._HA_activation == "standardize":
                    hop_a = (hop_a - hop_a.min(dim=2, keepdim=True)[0]) / (hop_a.max(dim=2, keepdim=True)[0] - hop_a.min(dim=2, keepdim=True)[0]).clamp(min=1e-9)

                hop_a = F.softmax(hop_a, dim=-1)
                # hop_a = self.attn_drop(hop_a)
                if not self.training:
                    self.hop_a = hop_a
                
                rst = 0
                for i in range(hop_a.shape[2]):
                    
                    if self._weight_style == "HA+HC":
                        rst += hstack[i] * hop_a[:, :, [i]] * self.weights[:, :, i, :]
                    else:
                        rst += hstack[i] * hop_a[:, :, [i]]

            if self._weight_style == "HC":
                rst = 0
                for i in range(len(hstack)):
                    rst += hstack[i] * self.weights[:, :, i, :]
            if self._weight_style == "mean":
                rst = 0
                for i in range(len(hstack)):
                    rst += hstack[i] / len(hstack)

            if self._propagate_first:
                rst = self.fc(rst)
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(feat).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

class AGDN(nn.Module):
    def __init__(
        self, 
        in_feats, 
        n_classes, 
        n_hidden, 
        n_layers, 
        n_heads, 
        activation, 
        K=3, 
        dropout=0.0, 
        input_drop=0.0, 
        edge_drop=0.0, 
        attn_drop=0.0, 
        diffusion_drop=0.0,
        use_attn_dst=True,
        position_emb=True,
        transition_matrix='gat_adj',
        weight_style="HA",
        HA_activation="leakyrelu",
        residual=True,
        bias_last=True,
        no_bias=False,
        zero_inits=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            # in_channels = n_heads if i > 0 else 1
            num_heads = n_heads if i < n_layers - 1 or n_layers == 1 else 1
            out_channels = n_heads

            self.convs.append(
                AGDNConv(
                    in_hidden, 
                    out_hidden, 
                    K=K, 
                    num_heads=num_heads, 
                    edge_drop=edge_drop, 
                    attn_drop=attn_drop, 
                    diffusion_drop=diffusion_drop,
                    use_attn_dst=use_attn_dst,
                    position_emb=position_emb,
                    transition_matrix=transition_matrix,
                    weight_style=weight_style,
                    HA_activation=HA_activation,
                    residual=residual,
                    bias=(not bias_last) and (not no_bias),
                    zero_inits=zero_inits,
                )
            )

            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        if bias_last and not no_bias:
            self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        else:
            self.bias_last = None

        self.input_dropout = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_dropout(h)
        h_last = h
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                if h_last.shape[-1] == h.shape[-1]:
                    h = h + h_last
                h = self.bns[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)
                h_last = h

        h = h.mean(1)
        if self.bias_last is not None:
            h = self.bias_last(h)

        return h
