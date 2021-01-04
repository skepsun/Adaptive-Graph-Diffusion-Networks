# adaptive_graph_diffusion_convolution_networks

The preprint paper is here: https://arxiv.org/abs/2012.15024

The framework of training and evaluating is adapted from [dgl's example codes]:(https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [Espylapiza's implementation]:(https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv).

To reproduce results of GCN-HA:

`python src/gcn-ha.py --seed 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --attn_drop 0.05 --epochs 2000 --n-runs 10 --use-labels`(with labels)

`python src/gcn-ha.py --seed 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --attn_drop 0.05 --epochs 2000 --n-runs 10` (without labels)

To reproduce results of GAT-HA:

`python src/gat-ha.py --seed 0 --n-label-iters 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --edge_drop 0.0 --attn_drop 0.05 --norm gcn --no-attn-dst --epochs 2000 --n-runs 10 --use-labels` (1 head with labels)

`python src/gat-ha.py --seed 0 --n-label-iters 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 1 --dropout 0.5 --input_drop 0.1 --edge_drop 0.0 --attn_drop 0.05 --norm gcn --no-attn-dst --epochs 2000 --n-runs 10` (1 head without labels)

`python src/gat-ha.py --seed 0 --n-label-iters 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.0 --attn_drop 0.05 --norm gcn --no-attn-dst --epochs 2000 --n-runs 10 --use-labels` (3 heads with labels)

`python src/gat-ha.py --seed 0 --n-label-iters 0 --lr 0.002 --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.0 --attn_drop 0.05 --norm gcn --no-attn-dst --epochs 2000 --n-runs 10` (3 heads without labels)

In GAT-HA, the argument "norm" means the way of extra normalization performed on "already" normalized GAT transition matrix. "norm==gcn" follows the operation from Espylapiza's implementation. "norm==gat" uses a geometric mean of attention scores normalized by "src" and by "dst". These two settings can be mixed, with "norm==both".

SGAT-HA is still in development and can be ignored for now.

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
