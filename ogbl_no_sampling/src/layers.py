from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch import nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax
import math
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning

class SAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    If a weight tensor on each edge is provided, the aggregation becomes:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that :math:`e_{ji}` is broadcastable with :math:`h_j^{l}`.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.

        SAGEConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import SAGEConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = SAGEConv(10, 2, 'pool')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 10)
    >>> conv = SAGEConv((5, 10), 2, 'mean')
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[ 0.3163,  3.1166],
            [ 0.3866,  2.5398],
            [ 0.5873,  1.6597],
            [-0.2502,  2.8068]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            # self.fc_self.reset_parameters()
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        # self.fc_neigh.reset_parameters()

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.q = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        # self.k = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=True)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != num_heads * out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
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
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # el = (feat_src * self.q(h_src).view(-1, self._num_heads, self._out_feats)).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.k(h_dst).view(-1, self._num_heads, self._out_feats)).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            a = edge_softmax(graph, e)
            if edge_feat is not None:
                graph.edata['w'] = edge_feat
                graph.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'w_d'))
                graph.apply_edges(fn.copy_u('w_d', 'w_src'))
                graph = graph.reverse(copy_edata=True)
                graph.apply_edges(fn.copy_u('w_d', 'w_dst'))
                graph = graph.reverse(copy_edata=True)
                graph.edata['w'] = graph.edata['w'] / torch.sqrt(graph.edata['w_src'] * graph.edata['w_dst'])
                a = (a + graph.edata['w'].unsqueeze(1)) / 2
            graph.edata['a'] = self.attn_drop(a)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class AGDNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 K,
                 feat_drop=0.,
                 attn_drop=0.,
                 diffusion_drop=0.,
                 edge_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 transition_matrix='gat',
                 weight_style="HA",
                 hop_norm=False,
                 pos_emb=True,
                 bias=True,
                 share_weights=True,
                 no_dst_attn=False,
                 pre_act=False):
        super(AGDNConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._K = K
        self._allow_zero_in_degree = allow_zero_in_degree
        self._transition_matrix = transition_matrix
        self._hop_norm = hop_norm
        self._weight_style = weight_style
        self._pos_emb = pos_emb
        self._share_weights = share_weights
        self._pre_act = pre_act
        self._edge_drop = edge_drop

        if residual:
            # if self._in_dst_feats != out_feats * num_heads:
            self.res_fc = nn.Linear(
                self._in_dst_feats, num_heads * out_feats, bias=False)
            # else:
            #     self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            if self._share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)
        
        if transition_matrix.startswith('gat'):
            if pre_act:
                self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            elif transition_matrix == 'gat_sym' or no_dst_attn:
                self.attn_l = self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            else:
                self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.diffusion_drop =nn.Dropout(diffusion_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if pos_emb:
            self.position_emb = nn.Parameter(torch.FloatTensor(size=(1, num_heads, K+1, out_feats)))
        if weight_style in ["HA", "HA+HC"]:
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(num_heads,)))
        if weight_style in ["HC", "HA+HC"]:
            self.weights = nn.Parameter(torch.ones(size=(1, num_heads, K+1, out_feats)))
        if weight_style == "lstm":
            self.lstm = nn.LSTM(self._out_feats, 1,
                                bidirectional=True, batch_first=True)
            self.att = Linear(2, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            # self.res_fc.reset_parameters()
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        # self.fc_src.reset_parameters()
        if not self._share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            # self.fc_dst.reset_parameters()
        if self._transition_matrix.startswith('gat'):
            if self._pre_act:
                nn.init.xavier_normal_(self.attn, gain=gain)
            else:
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._pos_emb:
            nn.init.xavier_normal_(self.position_emb, gain=gain)
        if self._weight_style in ["HA", "HA+HC"]:
            nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
            nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
            # nn.init.zeros_(self.hop_attn_l)
            # nn.init.zeros_(self.hop_attn_r)
            nn.init.uniform_(self.beta)

        elif self._weight_style in ["HC", "HA+HC"]:
            nn.init.xavier_uniform_(self.weights, gain=gain)
        elif self._weight_style == "lstm":
            self.lstm.reset_parameters()
            self.att.reset_parameters()
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
      

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def feat_trans(self, h, i):
        if self._hop_norm:
            h = F.normalize(h, dim=-1, p=2)
        # h = (h - h.mean(0)) / (h.std(0) + 1e-9)
        # h = (h-h.min(0)[0]) / (h.max(0)[0] - h.min(0)[0])
        if self._pos_emb:
            h = h + self.position_emb[:, :, i, :]
        # h = (0.5 ** i) * h
        return h

    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self._share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            
            graph.srcdata.update({'ft': feat_src})

            if self._transition_matrix.startswith('gat'):
                if self._pre_act:
                    graph.srcdata.update({'el': feat_src})
                    graph.dstdata.update({'er': feat_dst})
                else:
                    el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'el': el})
                    graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                # if edge_feat is not None:
                #     feat_e = self.fc_e(edge_feat).view(-1, self._num_heads, self._edge_feats)
                #     graph.edata['e'] = graph.edata['e'] + (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
                e = graph.edata.pop('e')
                e = self.leaky_relu(e)
                
                if self._pre_act:
                    e = (e * self.attn).sum(dim=-1).unsqueeze(-1)
                # if self.training and self._edge_drop > 0:
                #     perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                #     bound = int(graph.number_of_edges() * self._edge_drop)
                #     eids = perm[bound:]
                # else:
                #     eids = torch.arange(graph.number_of_edges(), device=graph.device)
                # compute softmax
                if self._transition_matrix == 'gat_sym':
                    a = torch.sqrt((edge_softmax(graph, e, norm_by='dst') * edge_softmax(graph, e, norm_by='src')).clamp(min=1e-9))
                elif self._transition_matrix == 'gat_col':
                    a = edge_softmax(graph, e, norm_by='src')
                else:
                    a = edge_softmax(graph, e, norm_by='dst')

                if edge_feat is not None:
                    graph.edata['w'] = edge_feat
                    graph.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'w_d'))
                    graph.apply_edges(fn.copy_u('w_d', 'w_src'))
                    graph = graph.reverse(copy_edata=True)
                    graph.apply_edges(fn.copy_u('w_d', 'w_dst'))
                    graph = graph.reverse(copy_edata=True)
                    graph.edata['w'] = graph.edata['w'] / torch.sqrt(graph.edata['w_src'] * graph.edata['w_dst'])
                    a = (a + graph.edata['w'].unsqueeze(1)) / 2
            elif self._transition_matrix == 'gcn':
                deg = graph.in_degrees()
                inv_deg = 1 / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                sqrt_inv_deg = inv_deg.pow(0.5)
                a = sqrt_inv_deg[graph.edges()[0]] * sqrt_inv_deg[graph.edges()[1]]
                a = a.view(-1, 1, 1)
            elif self._transition_matrix == 'row':
                deg = graph.in_degrees()
                inv_deg = 1. / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                a = inv_deg[graph.edges()[1]]
                a = a.view(-1, 1, 1)
            elif self._transition_matrix == 'col':
                deg = graph.out_degrees()
                inv_deg = 1. / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                a = inv_deg[graph.edges()[0]]
                a = a.view(-1, 1, 1)
                

            # message passing

            hstack = [self.feat_trans(graph.dstdata['ft'], 0)]
            h_query = self.feat_trans(graph.dstdata['ft'], 0).unsqueeze(2)
            
            for k in range(1, self._K+1):
                graph.ndata['ft'] = self.diffusion_drop(graph.ndata['ft'])
                graph.edata['a'] = self.attn_drop(a)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                fn.sum('m', 'ft'))
                hstack.append(self.feat_trans(graph.dstdata['ft'], k))
            hstack = torch.stack(hstack, dim=2)
            if self._weight_style in ["HC"]:
                rst = (hstack * self.attn_drop(self.weights)).sum(dim=2)
            elif self._weight_style in ["HA", "HA+HC"]:
                
                astack = (hstack * self.hop_attn_r.unsqueeze(2)).sum(dim=-1).unsqueeze(-1) \
                        + (h_query * self.hop_attn_l.unsqueeze(2)).sum(dim=-1).unsqueeze(-1)
                astack = self.leaky_relu(astack) 
                astack = F.softmax(astack, dim=2) * torch.exp(self.beta.view(1, -1, 1, 1))
                # astack = self.attn_drop(astack)
                if self._weight_style == "HA+HC":
                    hstack = hstack * self.weights
                rst = (hstack * astack).sum(dim=2)
            elif self._weight_style == "sum":
                rst = hstack.sum(dim=2)
            elif self._weight_style == "max_pool":
                rst = hstack.max(dim=2)[0]
            elif self._weight_style == "mean_pool":
                rst = hstack.mean(dim=2)
            elif self._weight_style == "lstm":
                alpha, _ = self.lstm(hstack.view(-1, self._K+1, self._out_feats))
                alpha = self.att(alpha)
                alpha = torch.softmax(alpha, dim=1)
                rst = (hstack * alpha.view(-1, self._num_heads, self._K+1, 1)).sum(dim=2)
            
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class MemAGDNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 K,
                 edge_feats=0,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 weight_style="HA",
                 bias=True,
                 blocks=1):
        super(MemAGDNConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self._K = K
        self._allow_zero_in_degree = allow_zero_in_degree
        self._weight_style = weight_style
        self._blocks = blocks
        self._in_src_feats_ = (self._in_src_feats + K - 1) // K
        self._in_dst_feats_ = (self._in_dst_feats + K - 1) // K
        self.fc = nn.Linear(
                self._in_src_feats_ * K, out_feats, bias=False)
        self.q = nn.Linear(self._in_src_feats_, out_feats, bias=False)
        self.k = nn.Linear(self._in_src_feats_, out_feats, bias=False)
        self.v = nn.Linear(self._in_src_feats_, self._in_src_feats_, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self._in_src_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self._in_dst_feats)))
        if edge_feats > 0:
            self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
            self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        else:
            self.attn_e = None
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.position_emb = nn.Parameter(torch.FloatTensor(size=(1, num_heads, K+1, self._in_src_feats_)))
        if weight_style == "HA":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self._out_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self._out_feats)))
        elif weight_style == "HC":
            self.weights = nn.Parameter(torch.ones(size=(1, num_heads, K+1, out_feats)))
        elif weight_style == "lstm":
            self.lstm = nn.LSTM(self._in_src_feats_, ((K+1) * self._in_src_feats_) // 2,
                                bidirectional=True, batch_first=True)
            self.att = Linear(2 * (((K+1)* self._in_src_feats_) // 2), 1)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            # if self._in_dst_feats != out_feats * num_heads:
            self.res_fc = nn.Linear(
                self._in_dst_feats, num_heads * out_feats, bias=False)
            # else:
            #     self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.attn_e is not None:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.position_emb, gain=gain)
        if self._weight_style == "HA":
            nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
            nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        elif self._weight_style == "HC":
            nn.init.xavier_normal_(self.weights, gain=gain)
        elif self._weight_style == "lstm":
            self.lstm.reset_parameters()
            self.att.reset_parameters()
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def feat_trans(self, h, i):
        # mean = h.mean(dim=-1).view(h.shape[0], self._num_heads, 1)
        # var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._num_heads, 1) + 1e-9
        # h = (h - mean) * self.scale[i] * torch.rsqrt(var) + self.offset[i]
        h = h + self.position_emb[:, :, i, :]
        return h

    def compute_attention(self, graph, feat, edge_feat):
        h_src = h_dst = self.feat_drop(feat)

        el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({ 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # if edge_feat is not None:
        #     feat_e = self.fc_e(edge_feat).view(-1, self._num_heads, self._edge_feats)
        #     graph.edata['e'] = graph.edata['e'] + (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        a = edge_softmax(graph, e, norm_by='dst')
        if edge_feat is not None:
            graph.edata['w'] = edge_feat
            graph.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'w_d'))
            graph.apply_edges(fn.copy_u('w_d', 'w_src'))
            graph = graph.reverse(copy_edata=True)
            graph.apply_edges(fn.copy_u('w_d', 'w_dst'))
            graph = graph.reverse(copy_edata=True)
            graph.edata['w'] = graph.edata['w'] / torch.sqrt(graph.edata['w_src'] * graph.edata['w_dst'])
            a = (a + graph.edata['w'].unsqueeze(1)) / 2
        return a

    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')


            xs_src = xs_dst = torch.chunk(torch.cat([feat.view(-1, 1, self._in_src_feats), torch.zeros(size=feat.shape[:-1]+(1,self._in_src_feats_*self._K-feat.size(-1),), device=feat.device)], dim=-1), self._K, dim=-1)
            x_0_src = 0
           
            for i in range(self._K):
                x_0_src += xs_src[i]
            
            xs_dst_new = xs_src_new = [self.feat_trans(x_0_src,0)]
            
            a = self.compute_attention(graph, feat.view(-1, 1, self._in_src_feats), edge_feat)
        
            for i in range(self._K):
                # message passing
                graph.ndata['ft'] = xs_dst_new[-1]
                graph.edata['a'] = self.attn_drop(a)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                fn.sum('m', 'ft'))
                xs_src_new.append(self.feat_trans(graph.ndata['ft'] + self.feat_drop(xs_src[i]),i+1))
            h = torch.cat(xs_src_new[1:], dim=-1)
            # hstack = torch.stack(xs_src_new[1:], dim=2)
            # hstack = self.fc(hstack)
            # Q = self.q(hstack)
            # K = self.k(hstack)
            # h = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(2,3)) / math.sqrt(self._out_feats), dim=-1), hstack)

            # h_query = hstack[:,:,[0],:]
            # if self._weight_style == "HC":
            #     rst = (hstack * self.weights).sum(dim=2) / self.weights.sum()
            # elif self._weight_style == "HA":
            #     astack = (hstack * self.hop_attn_r.unsqueeze(2)).sum(dim=-1).unsqueeze(-1) \
            #             + (h_query * self.hop_attn_l.unsqueeze(2)).sum(dim=-1).unsqueeze(-1)
            #     astack = self.leaky_relu(astack)
            #     # astack = F.relu(astack)
            #     astack = F.softmax(astack, dim=2)
            #     astack = self.attn_drop(astack)
            #     rst = (hstack * astack).sum(dim=2)
            # elif self._weight_style == "max_pool":
            #     rst = hstack.max(dim=2)[0]
            # elif self._weight_style == "mean_pool":
            #     rst = hstack.mean(dim=2)
            # elif self._weight_style == "lstm":
            #     alpha, _ = self.lstm(hstack)
            #     alpha = self.att(alpha).squeeze(-1)
            #     alpha = torch.softmax(alpha, dim=-1)
            #     rst = (hstack * alpha.unsqueeze(-1)).sum(dim=2)
            rst = self.fc(h)
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(feat).view(feat.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst