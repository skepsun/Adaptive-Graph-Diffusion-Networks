from multiprocessing import pool
import pdb

# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import dgl.function as fn
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from dgl.nn.pytorch import GraphConv, SAGEConv
from torch.nn import BatchNorm1d, Linear, Module, ModuleList, Parameter
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from layers import AGDNConv, GATConv


class GCN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, in_edge_feats=0, n_edge_hidden=8, bn=True, residual=True):
        super(GCN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = ModuleList()
        
        self.convs.append(GraphConv(in_feats, n_hidden, norm="both", allow_zero_in_degree=True))
        if bn:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(n_hidden))
        else:
            self.norms = None
        if in_edge_feats > 0:
            self.edge_encoders = ModuleList()
            self.edge_encoders.append(Linear(in_edge_feats, 1))
        else:
            self.edge_encoders = None
        for _ in range(n_layers - 2):
            self.convs.append(
                GraphConv(n_hidden, n_hidden, norm="both", allow_zero_in_degree=True))
            if bn:
                self.norms.append(BatchNorm1d(n_hidden))
            if in_edge_feats > 0:
                self.edge_encoders.append(Linear(in_edge_feats, 1))
        self.convs.append(GraphConv(n_hidden, out_feats, norm="both", allow_zero_in_degree=True))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, 1))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

        if self.edge_encoders is not None:
            for encoder in self.edge_encoders:
                encoder.reset_parameters()

    def forward(self, mfgs, feat, edge_feat=None):
        if not isinstance(mfgs, list):
            mfgs = [mfgs] * len(self.convs)
        h = F.dropout(feat, self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs):
            h_dst = h[:mfgs[i].num_dst_nodes()]
            h = conv(mfgs[i], h)
            if self.residual:
                if h_dst.shape[1] == h.shape[1]:
                    h = h + h_dst
            if i < len(self.convs) - 1:
                if self.norms is not None:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def inference(self, loader, x_all, device):
        pbar = tqdm(total=len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(len(self.convs)):
            xs = []
            for input_nodes, output_nodes, mfgs in loader:
                mfg = mfgs[0].to(device)
                x = x_all[input_nodes].to(device)
                x_dst = x[:output_nodes.size(0)]
                x = self.convs[i](mfg, (x, x_dst))
                if self.residual:
                    if x_dst.shape[1] == x.shape[1]:
                        x = x + x_dst
                if i != len(self.convs) - 1:
                    if self.norms is not None:
                        x = self.norms[i](x)
                    x = F.relu(x)
                xs.append(x.cpu())

            pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class GAT(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads,
                 dropout, input_drop, attn_drop, 
                 in_edge_feats=0, n_edge_hidden=8, 
                 bn=True, residual=True):
        super(GAT, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        if in_edge_feats > 0:
            self.edge_encoders = torch.nn.ModuleList()
        else:
            self.edge_encoders = None
        self.convs = ModuleList()
        
        self.convs.append(GATConv(in_feats, n_hidden, num_heads, attn_drop=attn_drop, edge_feats=n_edge_hidden, residual=True, allow_zero_in_degree=True))
        if bn:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
        else:
            self.norms = None
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(n_hidden * num_heads, n_hidden, num_heads, attn_drop=attn_drop, edge_feats=n_edge_hidden, residual=True, allow_zero_in_degree=True))
            if bn:
                self.norms.append(
                    BatchNorm1d(num_heads * n_hidden)
                )
            if in_edge_feats > 0:
                self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.convs.append(GATConv(n_hidden * num_heads, out_feats, num_heads, attn_drop=attn_drop, edge_feats=n_edge_hidden, residual=True, allow_zero_in_degree=True))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

        if self.edge_encoders is not None:
            for encoder in self.edge_encoders:
                encoder.reset_parameters()

    def forward(self, mfgs, feat, edge_feat=None):
        if not isinstance(mfgs, list):
            mfgs = [mfgs] * len(self.convs)
        h = F.dropout(feat, self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs):
            h_dst = h[:mfgs[i].num_dst_nodes()]
            if self.edge_encoders is not None:
                h_e = self.edge_encoders[i](mfgs[i].edata['feat'])
            else:
                h_e = None
            h = conv(mfgs[i], (h, h_dst), edge_feat=h_e).flatten(1)

            if self.residual:
                if h_dst.shape[-1] == h.shape[-1]:
                    h = h + h_dst
            
            if i < len(self.convs) - 1:
                if self.norms is not None:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def inference(self, loader, x_all, device):
        # pbar = tqdm(total=len(self.convs))
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(len(self.convs)):
            xs = []
            for input_nodes, output_nodes, mfgs in loader:
                mfg = mfgs[0].to(device)
                x = x_all[input_nodes].to(device)
                x_dst = x[:output_nodes.size(0)]
                if self.edge_encoders is not None:
                    h_e = self.edge_encoders[i](mfg.edata['feat'])
                else:
                    h_e = None
                x = self.convs[i](mfg, (x, x_dst), edge_feat=h_e).flatten(1)
                if self.residual:
                    if x_dst.shape[1] == x.shape[1]:
                        x = x + x_dst
                if i != len(self.convs) - 1:
                    if self.norms is not None:
                        x = self.norms[i](x)
                    x = F.relu(x)
                xs.append(x.cpu())

            # pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all

class SAGE(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, edge_feats=0, bn=True, residual=True):
        super(SAGE, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, n_hidden, 'mean'))
        if bn:
            self.norms = torch.nn.ModuleList()
            self.norms.append(BatchNorm1d(n_hidden))
        else:
            self.norms = None
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(n_hidden, n_hidden, 'mean'))
            if bn:
                self.norms.append(BatchNorm1d(n_hidden))
        self.convs.append(SAGEConv(n_hidden, out_feats, 'mean'))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, mfgs, feat, edge_feat=None):
        if not isinstance(mfgs, list):
            mfgs = [mfgs] * len(self.convs)
        h = F.dropout(feat, self.input_drop, training=self.training)

        for i, conv in enumerate(self.convs):
            h_dst = h[:mfgs[i].num_dst_nodes()]
            h = conv(mfgs[i], (h, h_dst))
            if self.residual:
                if h_dst.shape[1] == h.shape[1]:
                    h = h + h_dst
            if i < len(self.convs) - 1:
                if self.norms is not None:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def inference(self, loader, x_all, device):
        pbar = tqdm(total=len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(len(self.convs)):
            xs = []
            for input_nodes, output_nodes, mfgs in loader:
                mfg = mfgs[0].to(device)
                x = x_all[input_nodes].to(device)
                x_dst = x[:output_nodes.size(0)]
                x = self.convs[i](mfg, (x, x_dst))
                if self.residual:
                    if x_dst.shape[1] == x.shape[1]:
                        x = x + x_dst
                if i != len(self.convs) - 1:
                    if self.norms is not None:
                        x = self.norms[i](x)
                    x = F.relu(x)
                xs.append(x.cpu())

            pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class AGDN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads, K,
                 dropout, input_drop, attn_drop, 
                 in_edge_feats=0, n_edge_hidden=8, 
                 weight_style="HA", 
                 pos_emb=True,
                 residual=False, 
                 pooling=False):
        super(AGDN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.pooling = pooling
        if in_edge_feats > 0:
            self.edge_encoders = torch.nn.ModuleList()
        else:
            self.edge_encoders = None
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(AGDNConv(in_feats, n_hidden, num_heads, K, 
                        attn_drop=attn_drop, edge_feats=n_edge_hidden, 
                        allow_zero_in_degree=True,
                        weight_style=weight_style, 
                        pos_emb=pos_emb,
                        residual=False))
        self.norms.append(BatchNorm1d(num_heads * n_hidden))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        for _ in range(n_layers - 2):
            self.convs.append(
                AGDNConv(n_hidden * num_heads, n_hidden, num_heads, K, 
                    attn_drop=attn_drop, edge_feats=n_edge_hidden, 
                    allow_zero_in_degree=True,
                    weight_style=weight_style, 
                    pos_emb=pos_emb,
                    residual=False))
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
            if in_edge_feats > 0:
                self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.convs.append(AGDNConv(n_hidden * num_heads, out_feats, num_heads, K, 
                    attn_drop=attn_drop, edge_feats=n_edge_hidden, 
                    allow_zero_in_degree=True,
                    pos_emb=pos_emb,
                    weight_style=weight_style, residual=False))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()
        
        if self.edge_encoders is not None:
            for encoder in self.edge_encoders:
                encoder.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        
        h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            if self.edge_encoders is not None:
                h_e = F.relu(self.edge_encoders[i](edge_feat))
            else:
                h_e = None
            h = conv(graph, h, edge_feat=h_e).flatten(1)
            if self.residual:
                if h_last.shape[-1] == h.shape[-1]:
                    h += h_last
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_last = h
        if self.edge_encoders is not None:
            h_e = F.relu(self.edge_encoders[-1](edge_feat))
        else:
            h_e = None
        h = self.convs[-1](graph, h, edge_feat=h_e).flatten(1)
        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        if self.pooling:
            h = torch.cat([h, h.mean(0, keepdim=True)], dim=-1)
        return h


class DotPredictor(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, dropout):
        super().__init__()
        self.lins = ModuleList()
        self.lins.append(Linear(in_feats, n_hidden))
        for _ in range(n_layers - 2):
            self.lins.append(Linear(n_hidden, n_hidden))
        self.lins.append(Linear(n_hidden, out_feats))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            x = g.edata.pop('score')
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_feats, n_hidden))
        for _ in range(n_layers - 2):
            self.lins.append(torch.nn.Linear(n_hidden, n_hidden))
        self.lins.append(torch.nn.Linear(n_hidden, out_feats))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
