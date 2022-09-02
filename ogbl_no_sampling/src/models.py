import torch
from torch.nn import Module, ModuleList, Linear, Parameter, BatchNorm1d, LayerNorm
import torch.nn.functional as F


from dgl.nn.pytorch import GraphConv
from dgl import DropEdge
from layers import SAGEConv, GATConv, AGDNConv, MemAGDNConv

class GCN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, bn=True, residual=True):
        super(GCN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = ModuleList()
        if n_layers == 1:
            self.convs.append(GraphConv(in_feats, out_feats, norm='both', allow_zero_in_degree=True))
            self.norms = None
        else:
            self.convs.append(GraphConv(in_feats, n_hidden, norm="both", allow_zero_in_degree=True))
            if bn:
                self.norms = ModuleList()
                self.norms.append(BatchNorm1d(n_hidden))
            else:
                self.norms = None

        for _ in range(n_layers - 2):
            self.convs.append(
                GraphConv(n_hidden, n_hidden, norm="both", allow_zero_in_degree=True))
            if bn:
                self.norms.append(BatchNorm1d(n_hidden))

        if n_layers > 1:
            self.convs.append(GraphConv(n_hidden, out_feats, norm="both", allow_zero_in_degree=True))


        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_weight=edge_feat)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_last = h
        h = self.convs[-1](graph, h, edge_weight=edge_feat)
        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class SAGE(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, edge_feats=0, bn=True, residual=True):
        super(SAGE, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = torch.nn.ModuleList()
        if n_layers == 1:
            self.convs.append(SAGEConv(in_feats, out_feats, 'mean'))
            self.norms = None
        else:
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
        if n_layers > 1:
            self.convs.append(SAGEConv(n_hidden, out_feats, 'mean'))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_weight=edge_feat)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_last = h
        h = self.convs[-1](graph, h, edge_weight=edge_feat)
        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class GAT(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads,
                 dropout, input_drop, attn_drop, bn=True, residual=False):
        super(GAT, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = ModuleList()
        
        self.convs.append(GATConv(in_feats, n_hidden if n_layers > 1 else out_feats, num_heads, attn_drop=attn_drop, residual=True))
        if bn and n_layers > 1:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
        else:
            self.norms = None


        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(n_hidden * num_heads, n_hidden, num_heads, attn_drop=attn_drop, residual=True))
            if bn:
                self.norms.append(
                    BatchNorm1d(num_heads * n_hidden)
                )

        if n_layers > 1:
            self.convs.append(GATConv(n_hidden * num_heads, out_feats, num_heads, attn_drop=attn_drop, residual=True))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class AGDN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads, K,
                 dropout, input_drop, attn_drop, edge_drop, diffusion_drop,
                 transition_matrix='gat',
                 no_dst_attn=False,
                 weight_style="HA", bn=True, output_bn=False, hop_norm=False,
                 pos_emb=True, residual=False, share_weights=True, pre_act=False):
        super(AGDN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop

        self.convs = ModuleList()
        self.convs.append(AGDNConv(in_feats, n_hidden if n_layers > 1 else out_feats, num_heads, K, 
            attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop, 
            transition_matrix=transition_matrix, weight_style=weight_style, 
            no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
        if bn:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
        else:
            self.norms = None

        for _ in range(n_layers - 2):
            self.convs.append(
                AGDNConv(n_hidden * num_heads, n_hidden, num_heads, K, 
                    attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop,
                    transition_matrix=transition_matrix, weight_style=weight_style, 
                    no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
            if bn:
                self.norms.append(BatchNorm1d(num_heads * n_hidden))

        if n_layers > 1:
            self.convs.append(AGDNConv(n_hidden * num_heads, out_feats, num_heads, K, 
                attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop,
                transition_matrix=transition_matrix, weight_style=weight_style, 
                no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
            if bn and output_bn:
                self.norms.append(BatchNorm1d(n_hidden))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        if self.residual:
            h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            
            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h_last = h

        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        
        if self.norms is not None and len(self.norms) == len(self.convs):
            h = self.norms[-1](h)

        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        return h

class MemAGDN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads, K,
                 dropout, input_drop, attn_drop, in_edge_feats=0, n_edge_hidden=1, weight_style="HA", residual=False):
        super(MemAGDN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        if in_edge_feats > 0:
            self.edge_encoders = torch.nn.ModuleList()
        else:
            self.edge_encoders = None
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(MemAGDNConv(in_feats, n_hidden, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
        self.norms.append(BatchNorm1d(num_heads * n_hidden))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        for _ in range(n_layers - 2):
            self.convs.append(
                MemAGDNConv(n_hidden * num_heads, n_hidden, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
            if in_edge_feats > 0:
                self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.convs.append(MemAGDNConv(n_hidden * num_heads, out_feats, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.bias = Parameter(torch.FloatTensor(size=(1, out_feats)))
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
        torch.nn.init.zeros_(self.bias)

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        
        # h_last = h
        for i, conv in enumerate(self.convs[:-1]):

            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            # if self.residual:
            #     if h_last.shape[-1] == h.shape[-1]:
            #         h += h_last
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # h_last = h

        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        # if self.residual:
        #     if h_last.shape[1] == h.shape[1]:
        #         h = h + h_last
        h += self.bias
        return h

class DotPredictor(Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x

class CosPredictor(Module):
    def __init__(self):
        super(CosPredictor, self).__init__()

    def reset_parameters(self):
        return
    
    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1) / \
            torch.sqrt(torch.sum(x_i * x_i, dim=-1) * torch.sum(x_j * x_j, dim=-1)).clamp(min=1-9)
        return x
        
class LinkPredictor(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, bn=False):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        for i in range(n_layers):
            in_feats_ = in_feats if i == 0 else n_hidden
            out_feats_ = out_feats if i == n_layers - 1 else n_hidden
            self.lins.append(torch.nn.Linear(in_feats_, out_feats_))
            if bn and i < n_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_feats_))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        # x = torch.cat([x_i, x_j], dim=-1)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x