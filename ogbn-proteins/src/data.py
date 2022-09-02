import dgl.function as fn
import torch
import numpy as np
from sklearn import preprocessing
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

def preprocess(graph, labels, train_idx, n_classes):
    global n_node_feats

    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    le = preprocessing.LabelEncoder()
    species_unique = torch.unique(graph.ndata["species"])
    max_no = species_unique.max()
    le.fit(species_unique % max_no)
    species = le.transform(graph.ndata["species"].squeeze() % max_no)
    species = np.expand_dims(species, axis=1)

    enc = preprocessing.OneHotEncoder()
    enc.fit(species)
    one_hot_encoding = enc.transform(species).toarray()

    graph.ndata["x"] = torch.FloatTensor(one_hot_encoding)
    # graph.ndata["feat"] = torch.cat([one_hot_encoding, graph.ndata["feat"]], dim=1)

    # Only the labels in the training set are used as features, while others are filled with zeros.

    graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
    graph.ndata["train_labels_onehot"][train_idx] = labels[train_idx].float()
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    graph.create_formats_()

    deg_sqrt, deg_isqrt = compute_norm(graph)
    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_sqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_isqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))
    

    return graph, labels
