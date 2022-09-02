import math

import numpy as np
import torch
import dgl.function as fn
from utils import compute_norm
from torch_sparse import SparseTensor


class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:
            raise StopIteration() from None


class BatchSampler(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            shuf = torch.randperm(self.n).split(self.batch_size)
            for shuf_batch in shuf:
                yield shuf_batch
            yield None

class ShaDowKHopSampler(torch.utils.data.DataLoader):
    r"""The ShaDow :math:`k`-hop sampler from the `"Deep Graph Neural Networks
    with Shallow Subgraph Samplers" <https://arxiv.org/abs/2012.01380>`_ paper.
    Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs.
    A deep GNN on this local graph then smooths the informative local signals.
    Args:
        data (torch_geometric.data.Data): The graph data object.
        depth (int): The depth/number of hops of the localized subgraph.
        num_neighbors (int): The number of neighbors to sample for each node in
            each hop.
        node_idx (LongTensor or BoolTensor, optional): The nodes that should be
            considered for creating mini-batches.
            If set to :obj:`None`, all nodes will be
            considered.
        replace (bool, optional): If set to :obj:`True`, will sample neighbors
            with replacement. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
            :obj:`num_workers`.
    """
    def __init__(self, g, depth, num_neighbors,
                 node_idx, replace=False,
                 **kwargs):

        self.g = g
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx

        super().__init__(node_idx.tolist(), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, n_id):
        n_id = torch.tensor(n_id)

        row, col = self.g.edges()
        rowptr = torch.ops.torch_sparse.ind2ptr(row, self.g.number_of_nodes())
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr, col, n_id, self.depth, self.num_neighbors, self.replace)
        rowptr, col, n_id, e_id, ptr, root_n_id = out
        subg = self.g.subgraph(n_id)

        return n_id, root_n_id, subg

def random_partition(num_clusters, graph, shuffle=True):
    """random partition"""
    batch_size = int(math.ceil(graph.num_nodes() / num_clusters))
    perm = np.arange(0, graph.num_nodes())
    if shuffle:
        np.random.shuffle(perm)

    batch_no = 0

    while batch_no < graph.num_nodes():
        batch_nodes = perm[batch_no:batch_no + batch_size]
        batch_no += batch_size
        sub_g = graph.subgraph(batch_nodes)
        deg_sqrt, deg_isqrt = compute_norm(sub_g)
        sub_g.ndata["sub_deg"] = sub_g.in_degrees().clamp(min=1)
        sub_g.srcdata.update({"src_norm": deg_isqrt})
        sub_g.dstdata.update({"dst_norm": deg_sqrt})
        sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm_adjust"))

        sub_g.srcdata.update({"src_norm": deg_isqrt})
        sub_g.dstdata.update({"dst_norm": deg_isqrt})
        sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm"))
        sub_g.ndata["l"] = sub_g.ndata["train_labels_onehot"]
        for _ in range(9):
            sub_g.update_all(fn.copy_u("l", "m"), fn.mean("m", "l"))
        # eid = sub_g.edata[dgl.EID]
        # sub_g.edata["feat"] = graph.edata['feat'][eid]
        yield batch_nodes, sub_g

def random_partition_v2(num_clusters, graph, shuffle=True, save_e=[]):
    """random partition v2"""
    if shuffle:
        cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
    else:
        if not save_e:
            cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
            save_e.append(cluster_id)
        else:
            cluster_id = save_e[0]
#         assert cluster_id is not None   
    perm = np.arange(0, graph.num_nodes())
    batch_no = 0
    while batch_no < num_clusters:
        batch_nodes = perm[cluster_id == batch_no]
        batch_no += 1 
        sub_g = graph.subgraph(batch_nodes)
        # deg_sqrt, deg_isqrt = compute_norm(sub_g)
        # sub_g.ndata["sub_deg"] = sub_g.in_degrees().clamp(min=1)
        # sub_g.srcdata.update({"src_norm": deg_isqrt})
        # sub_g.dstdata.update({"dst_norm": deg_sqrt})
        # sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm_adjust"))

        # sub_g.srcdata.update({"src_norm": deg_isqrt})
        # sub_g.dstdata.update({"dst_norm": deg_isqrt})
        # sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm"))
        yield batch_nodes, sub_g

# Simple random node idx sampler,
# using the implementation of pyg-team/pytorch_geometric
class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, num_parts: int, shuffle: bool = False):
        self.N = num_nodes
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.randint(self.num_parts, (self.N, ), dtype=torch.long)
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts

# Simple random partition sampler based on random node sampler,
# adapted from the implementation of pyg-team/pytorch_geometric.
# But we found it slower than direct calling random_partition function
class RandomPartitionSampler(torch.utils.data.DataLoader):
    def __init__(self, g, num_parts: int, shuffle: bool = False, **kwargs):

        self.N = g.number_of_nodes()
        self.E = g.number_of_edges()

        self.g = g

        super().__init__(
            self, batch_size=1,
            sampler=RandomIndexSampler(self.N, num_parts, shuffle),
            collate_fn=self.__collate__, **kwargs)
        
    def __getitem__(self, idx):
        return idx
        
    def __collate__(self, n_id):
        n_id = n_id[0]
        subg = self.g.subgraph(n_id)
        return n_id, subg

