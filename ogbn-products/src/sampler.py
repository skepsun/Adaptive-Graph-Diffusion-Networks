import math

import dgl.function as fn
import numpy as np
from dgl.dataloading.base import (Sampler, set_edge_lazy_features,
                                  set_node_lazy_features)

from utils import compute_norm

# We offer 3 implementations of random partition

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
        deg_sqrt, deg_isqrt = compute_norm(sub_g)
        sub_g.ndata["sub_deg"] = sub_g.in_degrees().clamp(min=1)
        sub_g.srcdata.update({"src_norm": deg_isqrt})
        sub_g.dstdata.update({"dst_norm": deg_sqrt})
        sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm_adjust"))

        sub_g.srcdata.update({"src_norm": deg_isqrt})
        sub_g.dstdata.update({"dst_norm": deg_isqrt})
        sub_g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "sub_gcn_norm"))
        yield batch_nodes, sub_g


class RandomSampler(Sampler):
    """Random partition sampler

    For each call, the sampler samples a node subset and then returns a node induced subgraph.

    Parameters
    ----------
    prefetch_ndata : list[str], optional
        The node data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    prefetch_edata : list[str], optional
        The edge data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    output_device : device, optional
        The device of the output subgraphs.
    """
    def __init__(self, prefetch_ndata=None,
                 prefetch_edata=None, output_device='cpu'):
        super().__init__()
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device

    def sample(self, g, indices):
        """Sampling function

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor
            Placeholder not used.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        sg = g.subgraph(indices, relabel_nodes=True, output_device=self.output_device)
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return indices, sg

