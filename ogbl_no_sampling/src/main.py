import argparse
import math
import time
import dgl
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import numpy as np
import numpy_indexed as npi
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from dgl.sampling import random_walk
# from torch_cluster import random_walk
import os.path as osp
from gen_model import gen_model
from logger import Logger
from loss import calculate_loss
from utils import (RA_AA_CN, adjust_lr, count_parameters, evaluate_hits,
                   evaluate_mrr, filter_edge, precompute_adjs, seed,
                   to_undirected)


def compute_pred(h, predictor, edges, batch_size):
    preds = []
    for perm in DataLoader(range(edges.size(0)), batch_size):
        edge = edges[perm].t()

        preds += [predictor(h[edge[0]], h[edge[1]]).sigmoid().squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    return pred

def train_split(split_edge, device):
    source = split_edge['train']['source_node'].to(device)
    target = split_edge['train']['target_node'].to(device)
    pos_edge = torch.stack([source, target], dim=1)
    return pos_edge

def train(model, predictor, feat, edge_feat, graph, split_edge, optimizer, batch_size, args):
    model.train()
    predictor.train()

    if args.dataset == 'ogbl-citation2':
        pos_train_edge = train_split(split_edge, feat.device)
    else:
        pos_train_edge = split_edge['train']['edge'].to(feat.device)

    if 'weight' in split_edge['train']:
        edge_weight_margin = split_edge['train']['weight']
    else:
        edge_weight_margin = None
    if args.negative_sampler == 'strict_global':
        neg_train_edge = torch.randint(0, graph.number_of_nodes(), (int(1.1 * args.n_neg * pos_train_edge.size(0)),)+(2,), dtype=torch.long,
                                 device=feat.device)
        neg_train_edge = neg_train_edge[~graph.has_edges_between(neg_train_edge[:,0], neg_train_edge[:,1])]
        
        # neg_train_edge = negative_sampling(pos_train_edge.t(), graph.number_of_nodes(), 
        #                                     args.n_neg * len(pos_train_edge))
        neg_src = neg_train_edge[:, 0]
        neg_dst = neg_train_edge[:, 1]
        if neg_train_edge.size(0) < pos_train_edge.size(0) * args.n_neg:
            k = pos_train_edge.size(0) * args.n_neg - neg_train_edge.size(0)
            rand_index = torch.randperm(neg_train_edge.size(0))[:k]
            neg_src = torch.cat((neg_src, neg_src[rand_index]))
            neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
        else:
            neg_src = neg_src[:pos_train_edge.size(0) * args.n_neg]
            neg_dst = neg_dst[:pos_train_edge.size(0) * args.n_neg]
        neg_train_edge = torch.reshape(
                        torch.stack([neg_src, neg_dst], dim=1),
                        (-1, args.n_neg, 2))

    # idx = torch.rand(pos_train_edge.shape[0]) < 0.15
    # pos_train_edge = pos_train_edge[idx, :]
    # neg_train_edge = negative_sampling(torch.stack(list(graph.edges()), dim=0), num_nodes=graph.number_of_nodes(), num_neg_samples=pos_train_edge.shape[0])

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(graph, feat, edge_feat)
        

        edge = pos_train_edge[perm]

        pos_out = predictor(h[edge[:, 0]], h[edge[:, 1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        if args.negative_sampler == 'global':
            # Just do some trivial random sampling.
            neg_edge = torch.randint(0, graph.number_of_nodes(), (args.n_neg * edge.size(0),)+(2,), dtype=torch.long,
                                 device=h.device)
        elif args.negative_sampler == 'strict_global':
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2))
        else:
            dst_neg = torch.randint(0, graph.number_of_nodes(), (args.n_neg * edge.size(0),)+(1,), dtype=torch.long, device=h.device)
            neg_edge = torch.cat([edge[:,0].repeat(args.n_neg).unsqueeze(-1), dst_neg], dim=1)
        # edge = neg_train_edge[:, perm]

        neg_out = predictor(h[neg_edge[:,0]], h[neg_edge[:,1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # loss = pos_loss + neg_loss
        weight_margin = edge_weight_margin[perm].to(feat.device) if edge_weight_margin is not None else None

        loss = calculate_loss(pos_out, neg_out, args.n_neg, margin=weight_margin, loss_func_name=args.loss_func)
        # cross_out = predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)])
        # cross_loss = -torch.log(1 - cross_out.sigmoid() + 1e-15).sum()
        # loss = loss + 0.1 * cross_loss
        loss.backward()

        if args.clip_grad_norm > -1:
            if 'feat' not in graph.ndata:
                torch.nn.utils.clip_grad_norm_(feat, args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)
        

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

def test_split(split, split_edge, device):
    source = split_edge[split]['source_node'].to(device)
    target = split_edge[split]['target_node'].to(device)
    target_neg = split_edge[split]['target_node_neg'].to(device)
    pos_edge = torch.stack([source, target], dim=1)
    neg_edge = torch.stack([source.view(-1, 1).repeat(1, 1000).view(-1), target_neg.view(-1)], dim=1)
    return pos_edge, neg_edge

@torch.no_grad()
def test(model, predictor, feat, edge_feat, graph, full_edge_feat, full_graph, split_edge, evaluator, batch_size, args):
    model.eval()
    predictor.eval()
    if args.dataset == 'ogbl-citation2':
        pos_train_edge, neg_train_edge = test_split('eval_train', split_edge, feat.device)
        pos_valid_edge, neg_valid_edge = test_split('valid', split_edge, feat.device)
        pos_test_edge, neg_test_edge = test_split('test', split_edge, feat.device)
    else:
        pos_train_edge = split_edge['eval_train']['edge'].to(feat.device)
        pos_valid_edge = split_edge['valid']['edge'].to(feat.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(feat.device)
        pos_test_edge = split_edge['test']['edge'].to(feat.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(feat.device)


    h = model(graph, feat, edge_feat)
    pos_train_pred = compute_pred(h, predictor, pos_train_edge, batch_size)
    pos_valid_pred = compute_pred(h, predictor, pos_valid_edge, batch_size)
    neg_valid_pred = compute_pred(h, predictor, neg_valid_edge, batch_size)

    h = model(full_graph, feat, full_edge_feat)
    pos_test_pred = compute_pred(h, predictor, pos_test_edge, batch_size)
    neg_test_pred = compute_pred(h, predictor, neg_test_edge, batch_size)

    if args.eval_metric == 'hits':
        results = evaluate_hits(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'agdn', 'memagdn'])
    parser.add_argument('--clip-grad-norm', type=float, default=1)
    parser.add_argument('--use-valedges-as-input', action='store_true',
                        help='This option can only be used for ogbl-collab')
    parser.add_argument('--no-node-feat', action='store_true')
    parser.add_argument('--use-emb', action='store_true')
    parser.add_argument('--use-edge-feat', action='store_true')
    parser.add_argument('--train-on-subgraph', action='store_true')
    parser.add_argument('--year', type=int, default=0)

    
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--transition-matrix', type=str, default='gat')
    parser.add_argument('--hop-norm', action='store_true')
    parser.add_argument('--weight-style', type=str, default='HA', choices=['HC', 'HA', 'HA+HC', 'HA1', 'sum', 'max_pool', 'mean_pool', 'lstm'])
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--no-share-weights', action='store_true')
    parser.add_argument('--pre-act', action='store_true')
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--input-drop', type=float, default=0.)
    parser.add_argument('--edge-drop', type=float, default=0.)
    parser.add_argument('--attn-drop', type=float, default=0.)
    parser.add_argument('--diffusion-drop', type=float, default=0.)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--output-bn', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--no-dst-attn', action='store_true')
    
    parser.add_argument('--advanced-optimizer', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--negative-sampler', type=str, default='global', choices=['global', 'strict_global', 'persource'])
    parser.add_argument('--n-neg', type=int, default=1)
    parser.add_argument('--eval-metric', type=str, default='hits')
    parser.add_argument('--loss-func', type=str, default='CE')
    parser.add_argument('--predictor', type=str, default='MLP')

    
    parser.add_argument('--random_walk_augment', action='store_true')
    parser.add_argument('--walk_start_type', type=str, default='edge')
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--adjust-lr', action='store_true')

    parser.add_argument('--use-heuristic', action='store_true')
    parser.add_argument('--n-extra-edges', type=int, default=200000)
    parser.add_argument('--heuristic-method', type=str, default='CN')
    parser.add_argument('--extra-training-edges', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    device = torch.device(device)

    dataset = DglLinkPropPredDataset(name=args.dataset, root='/mnt/ssd/ssd/dataset')
    graph = dataset[0]

    if args.dataset in ['ogbl-citation2']:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
    if args.model in ['gcn']:
        graph = graph.remove_self_loop().add_self_loop()
    
    print(graph)

    has_edge_attr = len(graph.edata.keys()) > 0

    split_edge = dataset.get_edge_split()

    if 'weight' in graph.edata:
        graph.edata['weight'] = graph.edata['weight'].float()

    if 'year' in split_edge['train'].keys() and args.year > 0:
        mask = split_edge['train']['year'] >= args.year
        split_edge['train']['edge'] = split_edge['train']['edge'][mask]
        split_edge['train']['year'] = split_edge['train']['year'][mask]
        split_edge['train']['weight'] = split_edge['train']['weight'][mask]
        graph.remove_edges((graph.edata['year']<args.year).nonzero(as_tuple=False).view(-1))
        graph = to_undirected(graph)

    torch.manual_seed(12345)
    if args.dataset == 'ogbl-citation2':
        idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
        split_edge['eval_train'] = {
            'source_node': split_edge['train']['source_node'][idx],
            'target_node': split_edge['train']['target_node'][idx],
            'target_node_neg': split_edge['valid']['target_node_neg'],
        }
    else:
        idx = torch.randperm(split_edge['train']['edge'].size(0))
        idx = idx[:split_edge['valid']['edge'].size(0)]
        split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        # val_edge_index = split_edge['valid']['edge'].t()
        # full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        # data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        # data.full_adj_t = data.full_adj_t.to_symmetric()

        full_graph = graph.clone()
        # split_edge['valid']['year'] = split_edge['valid']['year'] - 1900

        full_graph.remove_edges(torch.arange(full_graph.number_of_edges()))
        full_graph.add_edges(split_edge['train']['edge'][:, 0], split_edge['train']['edge'][:, 1], 
                            {'weight': split_edge['train']['weight'].unsqueeze(1).float()})
        full_graph.add_edges(split_edge['valid']['edge'][:, 0], split_edge['valid']['edge'][:, 1],
                            {'weight': split_edge['valid']['weight'].unsqueeze(1).float()})
        full_graph = to_undirected(full_graph)

        # In official OGB example, use_valedges_as_input options only utilizes validation edges in inference.
        # However, as described in OGB rules, validation edges can also participate training after all hyper-parameters
        # are fixed. The suitable pipeline is: 1. Tune hyperparameters using validation set without touching it during training
        # and inference (except as targets). 2. Re-train final model using tuned hyperparemters using validation edges as input.
        split_edge['train']['edge'] = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0)
        split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=0)
        # mask = full_graph.edges()[0] < full_graph.edges()[1]
        # split_edge['train']['edge'] = torch.stack([full_graph.edges()[0][mask], full_graph.edges()[1][mask]], dim=1)
        # split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=0)
        # split_edge['train']['year'] = torch.cat([split_edge['train']['year'], split_edge['valid']['year']], dim=0)
    else:
        full_graph = graph
    
    if args.train_on_subgraph and 'year' in split_edge['train'].keys():
        mask = (graph.edata['year'] >= 2010).view(-1)
        
        filtered_nodes = torch.cat([graph.edges()[0][mask], graph.edges()[1][mask]], dim=0).unique()
        graph.remove_edges((~mask).nonzero(as_tuple=False).view(-1))
        
        split_edge['train'] = filter_edge(split_edge['train'], filtered_nodes)
        split_edge['valid'] = filter_edge(split_edge['valid'], filtered_nodes)
        # split_edge['test'] = filter_edge(split_edge['test'], filtered_nodes)
  

    edge_weight = graph.edata['weight'].view(-1).cpu().numpy() \
        if 'weight' in graph.edata.keys() else torch.ones(graph.number_of_edges())
    A = ssp.csr_matrix((edge_weight, (graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy())), 
                       shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    # multiplier = 1 / np.log(A.sum(axis=0))
    # multiplier[np.isinf(multiplier)] = 0
    # A_ = A.multiply(multiplier).tocsr()
    adjs = precompute_adjs(A)
    if args.use_heuristic:
        # We implement preliminary version of Edge Proposal Set: https://arxiv.org/abs/2106.15810
        method_dict = {'RA':0, 'AA':1, 'CN':2}
        target_idx = method_dict[args.heuristic_method]
        target_size = args.n_extra_edges
        if not osp.exists(f'../extra_edges/{args.dataset}.pt'):
            A2 = A @ A
            A2[A > 0] = 0
            row, col = A2.nonzero()
            row, col = row[~(row==col)], col[~(row==col)]
            extra_edges = torch.from_numpy(np.stack([row, col], axis=1)).long()
            print(f'Initial extra edge number: {len(extra_edges)}')
            # extra_edges = extra_edges[~(npi.in_(extra_edges, split_edge['train']['edge']) \
            #                                     | npi.in_(extra_edges[:, [1,0]], split_edge['train']['edge']))]
            if args.use_valedges_as_input:
                extra_edges = extra_edges[~(npi.in_(extra_edges, split_edge['valid']['edge']) \
                                                    | npi.in_(extra_edges[:, [1,0]], split_edge['valid']['edge']))]
            print(f'Additional edge number after filtering existing edges: {len(extra_edges)}')

            extra_scores = RA_AA_CN(adjs, extra_edges.t())
            
            torch.save([extra_edges, extra_scores], f'../extra_edges/{args.dataset}.pt')
        else:
            extra_edges, extra_scores = torch.load(f'../extra_edges/{args.dataset}.pt')
        _, idx = torch.sort(extra_scores[:, target_idx], descending=True)
        extra_edges = extra_edges[idx]
        extra_edges = extra_edges[:target_size]
        extra_scores = extra_scores[idx]
        extra_scores = extra_scores[:target_size, [target_idx]]
        print(extra_scores.max(), extra_scores.min())
        extra_scores = extra_scores.clamp(max=100)
        extra_scores = extra_scores / extra_scores.max()
        print(extra_scores.max(), extra_scores.min())
        # extra_scores = extra_scores / extra_scores.max()
        # extra_scores.clamp_(min=0.01)
        # graph.add_edges(extra_edges[:,0], extra_edges[:,1], {'weight': extra_scores})
        # full_graph.add_edges(extra_edges[:,0], extra_edges[:,1], {'weight': extra_scores})
        # graph = to_undirected(graph)
        # full_graph = to_undirected(full_graph)
        if args.extra_training_edges:
            split_edge['train']['edge'] = torch.cat([split_edge['train']['edge'], extra_edges], dim=0)
            if 'weight' in split_edge['train'].keys():
                split_edge['train']['weight'] = torch.ones_like(split_edge['train']['weight'])
                split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], extra_scores.view(-1,)], dim=0)


    full_edge_weight = full_graph.edata['weight'].view(-1).cpu().numpy() \
        if 'weight' in full_graph.edata.keys() else torch.ones(full_graph.number_of_edges())
    full_A = ssp.csr_matrix((full_edge_weight, (full_graph.edges()[0].cpu().numpy(), full_graph.edges()[1].cpu().numpy())), 
                       shape=(full_graph.number_of_nodes(), full_graph.number_of_nodes()))
    full_adjs = precompute_adjs(full_A)

    graph = graph.to(device)
    full_graph = full_graph.to(device)

    has_node_attr = 'feat' in graph.ndata
    if has_node_attr and (not args.use_emb) and (not args.no_node_feat):
        emb = None
        feat = graph.ndata['feat'].float()
    else:
        # Use learnable embedding if node attributes are not available
        n_heads = args.n_heads if args.model in ['gat', 'agdn'] else 1
        emb = torch.nn.Embedding(graph.number_of_nodes(), args.n_hidden).to(device)
        if not has_node_attr or args.no_node_feat:
            feat = emb.weight
        else:
            feat = torch.cat([graph.ndata['feat'].float(), emb.weight], dim=-1)

    # degs = graph.in_degrees()
    # inv_degs = 1. / degs
    # inv_degs[torch.isinf(inv_degs)] = 0
    # inv_log_degs = 1. / torch.log(degs)
    # inv_log_degs[torch.isinf(inv_log_degs)] = 0
    # deg_feat = torch.cat([degs.unsqueeze(-1), inv_degs.unsqueeze(-1), inv_log_degs.unsqueeze(-1)], dim=-1)
    # # deg_feat = (deg_feat - deg_feat.min(0)[0]) / (deg_feat.max(0)[0] - deg_feat.min(0)[0])
    # feat = feat * inv_log_degs.unsqueeze(-1)

    in_feats = feat.shape[1]

    if has_edge_attr and args.use_edge_feat:
        edge_feat = graph.edata['weight'].float()
        full_edge_feat = full_graph.edata['weight'].float()
        in_edge_feats = graph.edata['weight'].shape[1]
    else:
        edge_feat = None
        full_edge_feat = None
        in_edge_feats = 0

    evaluator = Evaluator(name=args.dataset)

    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }


        target_metrics = {'ogbl-collab': 'Hits@50',
                        'ogbl-ddi': 'Hits@20',
                        'ogbl-ppa': 'Hits@100'}

    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

        target_metrics = {'ogbl-citation2': 'MRR'}

    if args.random_walk_augment:
        if args.walk_start_type == 'edge':
            rw_start = torch.reshape(split_edge['train']['edge'], (-1,)).to(device)
        else:
            rw_start = torch.arange(0, graph.number_of_nodes(), dtype=torch.long).to(device)

    


    for run in range(args.runs):
        seed(args.seed + run)
        
        model, predictor = gen_model(args, in_feats, in_edge_feats, device)
        print(model)
        parameters = list(model.parameters()) + list(predictor.parameters())
        if emb is not None:
            parameters = parameters + list(emb.parameters())
            torch.nn.init.xavier_uniform_(emb.weight)
            num_param = count_parameters(model) + count_parameters(predictor) + count_parameters(emb)
        else:
            num_param = count_parameters(model) + count_parameters(predictor)
        print(f'Number of parameters: {num_param}')
        
        if args.advanced_optimizer:
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50 // args.eval_steps, verbose=True)
        else:
            optimizer = torch.optim.Adam(
                parameters,
                lr=args.lr)

        best_val = {}
        best_test = {}
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            if args.random_walk_augment:
                # Random walk augmentation from PLNLP repository
                # We add a restart prob to reduce sampled pairs
                # walk = random_walk(full_graph.edges()[0], full_graph.edges()[1], rw_start, walk_length=args.walk_length)
                walk, _ = random_walk(full_graph, rw_start, length=args.walk_length)
                pairs = []
                weights = []
                for j in range(args.walk_length):
                    pairs.append(walk[:, [0, j + 1]])
                    weights.append(torch.ones((walk.size(0),), dtype=torch.float) / (j + 1))
                    # weights.append(torch.ones((walk.size(0),), dtype=torch.float) *  math.exp(-(j + 1.) / 2))
                pairs = torch.cat(pairs, dim=0)
                weights = torch.cat(weights, dim=0)
                # remove self-loop edges
                mask = ((pairs[:, 0] - pairs[:, 1]) != 0) * (pairs[:, 1] != -1)
                
                split_edge['train']['edge'] = torch.masked_select(pairs, mask.view(-1, 1)).view(-1, 2)
                split_edge['train']['weight'] = torch.masked_select(weights, mask)
                # edges_and_weights = torch.cat([split_edge['train']['edge'], split_edge['train']['weight'].view(-1,1).to(device)], dim=1)
                # edges_and_weights = torch.unique(edges_and_weights, dim=0)
                # split_edge['train']['edge'] = edges_and_weights[:, :2].long().to(device)
                # split_edge['train']['weight'] = edges_and_weights[:, 2].cpu()
            
            loss = train(model, predictor, feat, full_edge_feat, full_graph, split_edge, optimizer,
                         args.batch_size, args)
            t2 = time.time()
            if epoch % args.eval_steps == 0:
                
                results = test(model, predictor, feat, edge_feat, graph, full_edge_feat, full_graph, split_edge, evaluator,
                               args.batch_size, args)
                t3 = time.time()
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    if key not in best_val:
                        best_val[key] = result[1]
                        best_test[key] = result[2]
                    elif result[1] > best_val[key]:
                        best_val[key] = result[1]
                        best_test[key] = result[2]
                
                if args.advanced_optimizer:
                    lr_scheduler.step(results[target_metrics[args.dataset]][1])

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Train/Val/Test: {100 * train_hits:.2f}/{100 * valid_hits:.2f}/{100 * test_hits:.2f}%, '
                              f'Best Val/Test: {100 * best_val[key]:.2f}/{100 * best_test[key]:.2f}%')
                    print(f'---Loss: {loss:.4f}---Train time: {(t2-t1):.4f}---Test time: {(t3-t2):.4f}---')
            if args.adjust_lr:
                adjust_lr(optimizer, epoch / args.epochs, args.lr)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
