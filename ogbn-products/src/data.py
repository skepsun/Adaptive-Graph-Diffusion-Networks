import dgl.function as fn
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils import compute_norm


def load_data(dataset, args):
    data = DglNodePropPredDataset(name=dataset, root=args.root)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    print(f"Nodes : {graph.number_of_nodes()}\n"
          f"Edges: {graph.number_of_edges()}\n"
          f"Train nodes: {len(train_idx)}\n"
          f"Val nodes: {len(val_idx)}\n"
          f"Test nodes: {len(test_idx)}")

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph, labels, train_idx, n_classes, args):
    n_classes = (labels.max() + 1).item()

    # graph = graph.remove_self_loop().add_self_loop()
    n_node_feats = graph.ndata["feat"].shape[-1]

    graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
    graph.ndata["train_labels_onehot"][train_idx, labels[train_idx, 0]] = 1
                
    graph.ndata["is_train"] = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
    graph.ndata["is_train"][train_idx] = 1
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)
    
    deg_sqrt, deg_isqrt = compute_norm(graph)
    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_sqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_isqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))


    graph.create_formats_()

    return graph, labels
