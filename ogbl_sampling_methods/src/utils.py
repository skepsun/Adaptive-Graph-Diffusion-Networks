import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import dgl


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def evaluate_hits(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results
        

def evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred=None):
    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred if neg_train_pred is None else neg_train_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)
    
    return results

def evaluate_rocauc(evaluator, pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    train_rocauc = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })[f'rocauc']
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'rocauc']
    test_rocauc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'rocauc']
    results['ROC-AUC'] = (train_rocauc, valid_rocauc, test_rocauc)
    return results

# def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
#     train_auc = roc_auc_score(train_true, train_pred)
#     valid_auc = roc_auc_score(val_true, val_pred)
#     test_auc = roc_auc_score(test_true, test_pred)
#     results = {}
#     results['AUC'] = (train_auc, valid_auc, test_auc)

#     return results

def precompute_adjs(A):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 0
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()
    return (A, D, D_log, D_common)


def RA_AA_CN(adjs, edge):
    A, D, D_log, D_common = adjs
    ra = []
    cn = []
    aa = []

    src, dst = edge
    ra = np.array(np.sum(A[src].multiply(D[dst]), 1))
    aa = np.array(np.sum(A[src].multiply(D_log[dst]), 1))
    cn = np.array(np.sum(A[src].multiply(D_common[dst]), 1))
        # break
    scores = np.concatenate([ra, aa, cn], axis=1)
    return torch.FloatTensor(scores)

def to_undirected(graph):
    print(f'Previous edge number: {graph.number_of_edges()}')
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
    keys = list(graph.edata.keys())
    for k in keys:
        if k != 'weight':
            graph.edata.pop(k)
        else:
            graph.edata[k] = graph.edata[k].float()
    graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True, aggregator='sum')
    print(f'After adding reversed edges: {graph.number_of_edges()}')
    return graph

def process_collab(graph, split_edge, args):
    if 'weight' in graph.edata:
        graph.edata['weight'] = graph.edata['weight'].float()

    if 'year' in split_edge['train'].keys() and args.year > 0:
        mask = split_edge['train']['year'] >= args.year
        split_edge['train']['edge'] = split_edge['train']['edge'][mask]
        split_edge['train']['year'] = split_edge['train']['year'][mask]
        split_edge['train']['weight'] = split_edge['train']['weight'][mask]
        graph.remove_edges((graph.edata['year']<args.year).nonzero(as_tuple=False).view(-1))
        graph = to_undirected(graph)
    
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
    
    # if args.train_on_subgraph and 'year' in split_edge['train'].keys():
    #     mask = (graph.edata['year'] >= 2010).view(-1)
        
    #     filtered_nodes = torch.cat([graph.edges()[0][mask], graph.edges()[1][mask]], dim=0).unique()
    #     graph.remove_edges((~mask).nonzero(as_tuple=False).view(-1))
        
    #     split_edge['train'] = filter_edge(split_edge['train'], filtered_nodes)
    #     split_edge['valid'] = filter_edge(split_edge['valid'], filtered_nodes)
    #     # split_edge['test'] = filter_edge(split_edge['test'], filtered_nodes)
    
    return full_graph, split_edge