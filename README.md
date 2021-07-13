# adaptive_graph_diffusion_convolution_networks

The preprint paper is here: https://arxiv.org/abs/2012.15024

The framework of training and evaluating is adapted from [dgl's example codes]:(https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [Espylapiza's implementation]:(https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv).

requirements: `torch, ogb, dgl, torch_geometric`

## ogbn-arxiv

To reproduce results of GCN-HA:

`python src/gcn-ha.py --seed 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --attn_drop 0.05 --epochs 2000 --n-runs 10 --use-labels`(with labels)

`python src/gcn-ha.py --seed 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --attn_drop 0.05 --epochs 2000 --n-runs 10` (without labels)

To reproduce results of GAT-HA, execute the `scripts`:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/{model}/{script}
```

In GAT-HA, the argument "norm" means the way of extra normalization performed on "already" normalized GAT transition matrix. "norm==sym" follows the form from Espylapiza's implementation. "norm==avg" uses a mean of GAT transition matrix and GCN transition matrix.


Results without using labels:
|  model   | test_acc  |
|  ----  | ----  |
| GCN-HA  | 73.24 ± 0.20 |
| GAT-HA_1_head  | 73.49 ± 0.15 |
| GAT-HA_3_heads | 73.75 ± 0.21 |

Results with using labels:
|  model   | test_acc  |
|  ----  | ----  |
| GCN-HA  | 73.39 ± 0.12 |
| GAT-HA_1_head  | 73.81 ± 0.13 |
| GAT-HA_3_heads | 73.98 ± 0.09 |

## ogbn-products
