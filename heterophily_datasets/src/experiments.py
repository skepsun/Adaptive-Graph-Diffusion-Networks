import argparse
import time
from tqdm import tqdm
import dgl
import dgl.function as fn
import numpy as np
from numpy.lib.function_base import append
import torch
import torch.nn.functional as F
import torch.nn as nn
from gen_model import gen_model
from tqdm import tqdm
from dgl.nn.pytorch.conv import SGConv
from models import AGDNConv
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import accuracy_score
import time
from utils import seed, compute_norm

class ACCEvaluator(object):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, pred_y, true_y):
        return accuracy_score(true_y.detach().cpu(), pred_y.detach().cpu())

class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, num_classes, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_feats, n_hidden)
        self.fc2 = nn.Linear(n_hidden, num_classes)
        self.bn1 = nn.BatchNorm1d(n_hidden)
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.bn1.reset_parameters()
    
    def forward(self, feat):
        h = self.fc1(feat)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return h

class SGC(nn.Module):
    def __init__(self, in_feats, n_hidden, num_classes, dropout, K=3):
        super(SGC, self).__init__()
        
        self.dropout = dropout
        self.K = K
        self.mlp = MLP(in_feats, n_hidden, num_classes, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, g, feat):
        g.ndata['ft'] = feat
        for k in range(self.K):
            g.update_all(fn.u_mul_e('ft', 'gcn_norm', 'm'), fn.sum('m', 'ft'))
        h = self.mlp(g.ndata['ft'])
        return h

class AGDN(nn.Module):
    def __init__(self, in_feats, n_hidden, num_classes, n_layers, n_heads, dropout, 
                K=3, transition_matrix='gcn', weight_style='HA', position_emb=True,
                residual=False):
        super(AGDN, self).__init__()
        
        self.dropout = dropout
        self.bns = nn.ModuleList()
        self.K = K
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_feats_ = in_feats if i == 0 else n_hidden * n_heads
            out_feats_ = n_hidden if i < n_layers - 1 else num_classes
            self.convs.append(AGDNConv(in_feats_, out_feats_, num_heads=n_heads,
                K=K, weight_style=weight_style, transition_matrix=transition_matrix,
                position_emb=position_emb, zero_inits=True, residual=residual))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_feats_ * n_heads))
        self.reset_parameters()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, feat):
        h = feat
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            if i < len(self.convs) - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = h.mean(1)
        return h

def count_parameters(model):
    # print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def train(model, g, feat, labels, train_mask, loss_func, optimizer, device):
    model.train()
    out = model(g, feat)
    loss = loss_func(out[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, g, feat, labels, train_mask, val_mask, test_mask, loss_func, evaluator, device):
    model.eval()
    out = model(g, feat)
    pred = out.argmax(-1)
    train_loss = loss_func(out[train_mask], labels[train_mask]).item()
    val_loss = loss_func(out[val_mask], labels[val_mask]).item()
    test_loss = loss_func(out[test_mask], labels[test_mask]).item()

    train_score = evaluator(pred[train_mask], labels[train_mask])
    val_score = evaluator(pred[val_mask], labels[val_mask])
    test_score = evaluator(pred[test_mask], labels[test_mask])

    return out, train_loss, val_loss, test_loss, train_score, val_score, test_score

def run(g, feat, labels, train_mask, val_mask, test_mask, evaluator, device, args):
    in_feats = feat.shape[1]
    n_classes = len(labels.unique())
    model = AGDN(in_feats, args.n_hidden, n_classes, args.n_layers,
                args.n_heads,
                args.dropout, 
                K = args.K, 
                transition_matrix=args.transition_matrix,
                weight_style=args.weight_style,
                position_emb=args.pos_emb,
                residual=args.residual)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_func = torch.nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_val_score = 0
    final_test_score = 0
    best_epoch = 0
    for epoch in range(1, args.n_epochs+1):
        tic = time.time()
        train(model, g, feat, labels, train_mask, loss_func, optimizer, device)
        if epoch % args.eval_steps == 0:
            out, train_loss, val_loss, test_loss, train_score, val_score, test_score \
                = test(model, g, feat, labels, train_mask, val_mask, test_mask, loss_func, evaluator, device)
            toc = time.time()
            if best_val_score < val_score:
                best_val_loss = val_loss
                best_val_score = val_score
                final_test_score = test_score
                best_epoch = epoch
                best_hop_weights = []
                if args.weight_style == "HA":
                    # best_hop_weights.append(model.conv.hop_a.detach().cpu().numpy())
                    for conv in model.convs:
                        best_hop_weights.append(conv.hop_a.detach().cpu().numpy())
                if args.weight_style == "HC":
                    # best_hop_weights.append(model.conv.weights.detach().cpu().numpy())
                    for conv in model.convs:
                        best_hop_weights.append(conv.weights.detach().cpu().numpy())
    
            # if epoch % log_steps == 0:
            #     print(f'epoch: {epoch}, time: {(toc-tic):.4f}s, loss: {train_loss:.4f}, score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}, best scores: {best_val_score:.4f}/{final_test_score:.4f}')

    return model, best_val_loss, best_val_score, final_test_score, best_epoch, best_hop_weights

def experiment(g, feat, labels, train_mask, val_mask, test_mask, evaluator, device, args):
    best_val_scores_ = []
    final_test_scores_ = []
    hop_weights_list_ = []
    for K in range(1, 9):
        args.K = K
        best_val_scores = []
        final_test_scores = []
        hop_weights_list = []
        bar = tqdm(range(5))
        for i in bar:
            seed(i)
            model, best_val_loss, best_val_score, final_test_score, best_epoch, best_hop_weights = \
                run(g, feat, labels, train_mask, val_mask, test_mask, evaluator, device, args)
                    
            # print(f'{weight_style}, run {i}, K {K}: {final_test_score:.4f}, best_epoch: {best_epoch}')
            best_val_scores.append(best_val_score)
            final_test_scores.append(final_test_score)
            hop_weights_list.append(best_hop_weights)
            bar.set_description(\
                f'{best_val_score:.4f}/{final_test_score:.4f}, {np.mean(best_val_scores):.4f}±{np.std(best_val_scores):.4f}/{np.mean(final_test_scores):.4f}±{np.std(final_test_scores):.4f}')
        bar.set_description(f'params: {count_parameters(model)}, {args.weight_style}, pos_emb {args.pos_emb}, K {args.K}: {np.mean(best_val_scores):.4f}±{np.std(best_val_scores):.4f}/{np.mean(final_test_scores):.4f}±{np.std(final_test_scores):.4f}')
        bar.close()
        best_val_scores_.append(best_val_scores)
        final_test_scores_.append(final_test_scores)
        hop_weights_list_.append(hop_weights_list)
    return best_val_scores_, final_test_scores_, hop_weights_list_

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--n-hidden', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-style', type=str, default='HA')
    parser.add_argument('--transition-matrix', type=str, default='gcn')
    parser.add_argument('--no-position-emb', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--eval-steps', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=100)
    return parser.parse_args()

def main(args):
    # DataLoading
    dataset_name = 'ogbn-arxiv'
    root = '/mnt/ssd/ssd/dataset'

    dataset = DglNodePropPredDataset(name=dataset_name, root=root)
    g, labels = dataset[0]
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    labels = labels.squeeze(1)

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    g = g.to(device)
    labels = labels.to(device)

    deg_sqrt, deg_isqrt = compute_norm(g)
        
        
    g.srcdata.update({"src_norm": deg_isqrt})
    g.dstdata.update({"dst_norm": deg_isqrt})
    g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    g.srcdata.update({"src_norm": deg_isqrt})
    g.dstdata.update({"dst_norm": deg_sqrt})
    g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))
    evaluator = ACCEvaluator()

    feat = g.ndata['feat']
    # labels = g.ndata['label']
    in_feats = feat.shape[1]
    n_classes = len(labels.unique())

    train_idx, val_idx, test_idx = dataset.get_idx_split()['train'], dataset.get_idx_split()['valid'], dataset.get_idx_split()['test']
    train_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    test_mask[test_idx] = True

    hop_weights_dict = {}
    for weight_style in ['mean', 'HA', 'HC']:
        args.weight_style = weight_style
        best_val_scores, final_test_scores, hop_weights_list = experiment(g, feat, labels, train_mask, val_mask, test_mask, evaluator, device, args)
        hop_weights_dict[args.weight_style] = hop_weights_list
    
    torch.save(hop_weights_dict, f'{args.transition_matrix}_hop_weights.pt')
