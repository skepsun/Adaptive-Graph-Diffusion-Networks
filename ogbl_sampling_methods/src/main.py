import argparse
import pdb
import time
from os import device_encoding
from posixpath import split

import dgl
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from dgl.dataloading import (ClusterGCNSampler, EdgeDataLoader,
                             MultiLayerFullNeighborSampler,
                             MultiLayerNeighborSampler, NodeDataLoader,
                             SAINTSampler, ShaDowKHopSampler)
from dgl.dataloading.negative_sampler import GlobalUniform, PerSourceUniform
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_model import gen_model
from logger import Logger
from loss import calculate_loss
from utils import RA_AA_CN, evaluate_hits, evaluate_mrr, precompute_adjs, seed, count_parameters, process_collab


def compute_pred(h, predictor, edges, batch_size, device):
    preds = []
    # pbar = tqdm(total=edges.size(0))
    # pbar.set_description('Evaluating')
    for perm in DataLoader(range(edges.size(0)), batch_size):
        edge = edges[perm]
        preds += [predictor(h[edge[:, 0]].to(device), h[edge[:, 1]].to(device)).sigmoid().squeeze().cpu()]
    #     pbar.update(len(perm))
    # pbar.close()
    # pbar.clear()
    pred = torch.cat(preds, dim=0)
    return pred

def train(model, predictor, feat, edge_feat, graph, split_edge, dataloader, optimizer, batch_size, device, args):
    model.train()
    predictor.train()

    # idx = torch.rand(pos_train_edge.shape[0]) < 0.15
    # pos_train_edge = pos_train_edge[idx, :]
    # neg_train_edge = negative_sampling(torch.stack(list(graph.edges()), dim=0), num_nodes=graph.number_of_nodes(), num_neg_samples=pos_train_edge.shape[0])
    # pbar = tqdm(total=1000)
    # pbar.set_description('Training')
    total_loss = total_examples = 0
    
    if args.sampler in ['neighborsampler', 'shadow']:
        for i, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(dataloader):
            if not isinstance(mfgs, list):
                mfgs = mfgs.to(device)
            else:
                mfgs = [mfg.to(device) for mfg in mfgs]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            pos_edge_src, pos_edge_dst = pos_graph.edges()
            neg_edge_src, neg_edge_dst = neg_graph.edges()
            inputs = feat[input_nodes].to(device)

            pos_score = predictor(outputs[pos_edge_src], outputs[pos_edge_dst])
            neg_score = predictor(outputs[neg_edge_src], outputs[neg_edge_dst])

            outputs = model(mfgs, inputs)
            if 'weight' in pos_graph.edata:
                weight_margin = pos_graph.edata['weight']
            else:
                weight_margin = None

            loss = calculate_loss(pos_score, neg_score, args.n_neg, margin=weight_margin, loss_func_name=args.loss_func)
            # score = torch.cat([pos_score, neg_score])
            # label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            # loss = F.binary_cross_entropy_with_logits(score, label)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)
            optimizer.step()

            num_examples = pos_score.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            # pbar.update(1)

            # We don't run a full epoch here since that just takes too much time.
            if (i + 1) % (1000) == 0:
                break
    
    if args.sampler in ['clustergcn', 'saint_node', 'saint_edge', 'saint_rw']:
        for subgraph in dataloader:
            subgraph = subgraph.to(device)
            optimizer.zero_grad()
            h = model(subgraph, subgraph.ndata['feat'])

            src, dst = subgraph.edges()
            # mask = src != dst
            # src, dst = src[mask], dst[mask]
            pos_out = predictor(h[src], h[dst])
            # pos_loss = -torch.log(pos_out + 1e-15).mean()

            # Just do some trivial random sampling.
            if args.negative_sampler == 'global':
                src_neg = torch.randint(0, subgraph.number_of_nodes(), (args.n_neg * src.size(0),) + src.size()[1:],
                                    dtype=torch.long, device=device)
            else:
                src_neg = src.repeat(args.n_neg)

            dst_neg = torch.randint(0, subgraph.number_of_nodes(), (args.n_neg * src.size(0),) + src.size()[1:],
                                    dtype=torch.long, device=device)
            neg_out = predictor(h[src_neg], h[dst_neg])
            # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            # loss = pos_loss + neg_loss
            if 'weight' in subgraph.edata:
                weight_margin = subgraph.edata['weight']
            else:
                weight_margin = None

            loss = calculate_loss(pos_out, neg_out, args.n_neg, margin=weight_margin, loss_func_name=args.loss_func)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)
            optimizer.step()

            num_examples = src.size(0)
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
def test(model, predictor, feat, edge_feat, graph, split_edge, dataloader, evaluator, batch_size, device, eval_device, args):
    model.eval()
    predictor.eval()
    if args.dataset == 'ogbl-citation2':
        pos_train_edge, neg_train_edge = test_split('eval_train', split_edge, feat.device)
        pos_valid_edge, neg_valid_edge = test_split('valid', split_edge, feat.device)
        pos_test_edge, neg_test_edge = test_split('test', split_edge, feat.device)
    else:
        pos_train_edge = split_edge['eval_train']['edge'].to(feat.device)
        neg_train_edge = None
        pos_valid_edge = split_edge['valid']['edge'].to(feat.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(feat.device)
        pos_test_edge = split_edge['test']['edge'].to(feat.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(feat.device)

    # if args.sampler == 'neighborsampler':
    #     h = model.inference(dataloader, feat, device)
    
    if args.sampler in ['neighborsampler', 'shadow', 'clustergcn', 'saint_node', 'saint_edge', 'saint_rw']:
        model = model.to(eval_device)
        start = time.time()
        h = model(graph.to(eval_device), feat.to(eval_device)).to(device)
        model = model.to(device)
        end = time.time()
        # print(f"Evaluating on CPU costs {(end-start):.2f}s")
    else:
        h = []
        for input_nodes, output_nodes, subgraph in dataloader:
            subgraph = subgraph.to(device)
            h.append(model(subgraph, subgraph.ndata['feat'])[:len(output_nodes)].to(device))
        h = torch.cat(h, dim=0)
    pos_train_pred = compute_pred(h, predictor, pos_train_edge, batch_size, device)
    if neg_train_edge is not None:
        neg_train_pred = compute_pred(h, predictor, neg_train_edge, batch_size, device)
        if args.dataset == 'ogbl-citation2':
            neg_train_pred = neg_train_pred.view(-1, 1000)
    else:
        neg_train_pred = None
    pos_valid_pred = compute_pred(h, predictor, pos_valid_edge, batch_size, device)
    neg_valid_pred = compute_pred(h, predictor, neg_valid_edge, batch_size, device)
    if args.dataset == 'ogbl-citation2':
        neg_valid_pred = neg_valid_pred.view(-1, 1000)


    pos_test_pred = compute_pred(h, predictor, pos_test_edge, batch_size, device)
    neg_test_pred = compute_pred(h, predictor, neg_test_edge, batch_size, device)
    if args.dataset == 'ogbl-citation2':
        neg_test_pred = neg_test_pred.view(-1, 1000)

    if args.eval_metric == 'hits':
        results = evaluate_hits(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred=neg_train_pred)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval-device', type=int, default=-1)
    parser.add_argument('--log-steps', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbl-citation2')
    parser.add_argument('--eval-metric', type=str, default='mrr')
    parser.add_argument('--use-valedges-as-input', action='store_true',
                        help='This option can only be used for ogbl-collab')
    parser.add_argument('--year', type=int, default=0)
    parser.add_argument('--no-node-feat', action='store_true')
    parser.add_argument('--use-emb', action='store_true')
    parser.add_argument('--loss-func', type=str, default='CE')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--eval-from', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-batch-size', type=int, default=64 * 1024)
    parser.add_argument('--eval-node-batch-size', type=int, default=2000)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--clip-grad-norm', type=int, default=0,
                        help='We do not clip gradient norm when this option is set <= 0')

    parser.add_argument('--sampler', type=str, default='shadow', choices=['neighborsampler', 'shadow', 'clustergcn', 'saint_node', 'saint_edge', 'saint_rw'])
    parser.add_argument('--negative-sampler', type=str, default='global', choices=['persource', 'global'])
    parser.add_argument('--n-neg', type=int, default=1)
    parser.add_argument('--n-clusters', type=int, default=15000, 
                        help='The cluster number for clustergcn sampler.')
    parser.add_argument('--node-budget', type=int, default=80000)
    parser.add_argument('--edge-budget', type=int, default=30000)
    parser.add_argument('--rw-budget', type=int, nargs='+', default=[20000, 3])
    parser.add_argument('--n-subgraphs', type=int, default=1000,
                        help='The subgraph number for graphsaint sampler')
    parser.add_argument('--neighbor-size', type=int, nargs='+', default=[15,10,5],
                        help='The neighbor size of each hop for neighborsampler/shadowkhopsampler')

    parser.add_argument('--model', type=str, default='gat', choices=['gcn', 'gat', 'sage', 'agdn'])
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-hidden', type=int, default=256)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--residual', action='store_true')

    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--input-drop', type=float, default=0.)
    parser.add_argument('--attn-drop', type=float, default=0.)

    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--weight-style', type=str, default='HA', choices=['HC', 'HA'])
    
    
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    device = torch.device(device)
    eval_device = f'cuda:{args.eval_device}' if args.eval_device > -1 else 'cpu'
    eval_device = torch.device(eval_device)

    dataset = DglLinkPropPredDataset(name=args.dataset, root='/mnt/ssd/ssd/dataset')
    graph = dataset[0]
    if args.dataset in ['ogbl-citation2', 'ogbl-ppa']:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)
    print(graph)
    # graph.edata['year'] = (graph.edata['year'] - 1950) / 100
    has_edge_attr = 'weight' in graph.edata.keys()
    # if has_edge_attr:
    #     train_feat = []
    #     for k, v in graph.edata.items():
    #         if 'year' in k:
    #             v = (v - 1900)/10
    #         if 'edge' not in k:
    #             train_feat.append(v.unsqueeze(-1) if len(v.shape) == 1 else v)
        
    #     graph.edata['feat'] = torch.cat(train_feat, dim=-1)
    

    split_edge = dataset.get_edge_split()
    if args.dataset == 'ogbl-collab':
        grpah, split_edge = process_collab(graph, split_edge, args)

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

    # graph = graph.to(device)

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

    in_feats = feat.shape[1]

    if has_edge_attr:
        edge_feat = graph.edata['weight']
        in_edge_feats = graph.edata['weight'].shape[1] if args.use_edge_feat else 0
    else:
        edge_feat = None
        in_edge_feats = 0

    if args.negative_sampler == 'persource':
        negative_sampler = PerSourceUniform(1)
    
    if args.negative_sampler == 'global':
        negative_sampler = GlobalUniform(1)
    
    if args.sampler == 'neighborsampler':
        train_sampler = MultiLayerNeighborSampler([15, 10, 5], replace=False)
        train_sampler = dgl.dataloading.as_edge_prediction_sampler(
            train_sampler, negative_sampler=negative_sampler)
        train_dataloader = dgl.dataloading.DataLoader(
            # The following arguments are specific to NodeDataLoader.
            graph,                                  # The graph
            torch.arange(graph.number_of_edges()),  # The edges to iterate over
            train_sampler,                                # The neighbor sampler
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=args.batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=args.n_workers     # Number of sampler processes
        )

        # eval_sampler = MultiLayerNeighborSampler([30] * args.n_layers)
        eval_sampler = MultiLayerFullNeighborSampler(1)
        eval_dataloader = dgl.dataloading.DataLoader(
            graph, torch.arange(graph.number_of_nodes()), eval_sampler,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.n_workers,
            device='cpu')
    
    if args.sampler == "shadow":
        train_sampler = ShaDowKHopSampler(args.neighbor_size)
        train_sampler = dgl.dataloading.as_edge_prediction_sampler(
            train_sampler, negative_sampler=negative_sampler)
        train_dataloader = dgl.dataloading.DataLoader(
            graph, graph.edges(form='eid'), train_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=args.n_workers)
        eval_sampler = ShaDowKHopSampler(args.neighbor_size)
        eval_dataloader = NodeDataLoader(graph, torch.arange(graph.number_of_nodes()), eval_sampler, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=args.n_workers)
    
    if args.sampler in ['clustergcn']:
        train_sampler = ClusterGCNSampler(graph, args.n_clusters)
        train_dataloader = dgl.dataloading.DataLoader(
            graph, torch.arange(args.n_subgraphs), train_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=args.n_workers)
        eval_sampler = ShaDowKHopSampler(args.batch_size)
        eval_dataloader = dgl.dataloading.DataLoader(graph, torch.arange(graph.number_of_nodes()), eval_sampler, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=args.n_workers)

    if args.sampler in ['saint_node', 'saint_edge', 'saint_rw']:
        if args.sampler == 'saint_node':
            mode = 'node'
            budget = args.node_budget
        if args.sampler == 'saint_edge':
            mode = 'edge'
            budget = args.edge_budget
        if args.sampler == 'saint_rw':
            mode = 'walk'
            budget = args.rw_budget
        train_sampler = SAINTSampler(mode, budget)
        train_dataloader = dgl.dataloading.DataLoader(
            graph, torch.arange(args.n_subgraphs), train_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=args.n_workers)
        eval_sampler = ShaDowKHopSampler(args.batch_size)
        eval_dataloader = dgl.dataloading.DataLoader(graph, torch.arange(graph.number_of_nodes()), eval_sampler, 
                                                batch_size=args.eval_batch_size, 
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=args.n_workers)

    evaluator = Evaluator(name=args.dataset)
    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

    for run in range(args.runs):
        seed(args.seed + run)
        model, predictor = gen_model(args, in_feats, in_edge_feats, device)
        parameters = list(model.parameters()) + list(predictor.parameters())
        if emb is not None:
            parameters = parameters + list(emb.parameters())
            torch.nn.init.xavier_uniform_(emb.weight)
            num_param = count_parameters(model) + count_parameters(predictor) + count_parameters(emb)
        else:
            num_param = count_parameters(model) + count_parameters(predictor)
        print(f'Number of parameters: {num_param}')
        # optimizer = torch.optim.Adam(
        #     parameters,
        #     lr=args.lr)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=100, verbose=True)
        best_val = {}
        best_test = {}
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, feat, edge_feat, graph, split_edge, train_dataloader, optimizer,
                         args.batch_size, device, args)
            lr_scheduler.step(loss)
            t2 = time.time()
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Time: {t2-t1:.4f}')
            
            if epoch >= args.eval_from and (epoch % args.eval_steps == 0):
                
                results = test(model, predictor, feat, edge_feat, graph, split_edge, eval_dataloader, evaluator,
                               args.eval_batch_size, device, eval_device, args)
                t3 = time.time()
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    if key not in best_val:
                        best_val[key] = result[1]
                        best_test[key] = result[2]
                    elif result[1] > best_val[key]:
                        best_val[key] = result[1]
                        best_test[key] = result[2]

                if epoch >= args.eval_from and (epoch % args.log_steps == 0):
                    for key, result in results.items():
                        train_scores, valid_scores, test_scores = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Train/Val/Test: {train_scores:.4f}/{valid_scores:.4f}/{test_scores:.4f}, '
                              f'Best Val/Test: {best_val[key]:.4f}/{best_test[key]:.4f}')
                    print(f'---Loss: {loss:.4f}---Train time: {(t2-t1):.4f}---Test time: {(t3-t2):.4f}---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
