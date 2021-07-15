# adaptive_graph_diffusion_convolution_networks

The preprint paper is here: https://arxiv.org/abs/2012.15024

The framework of training and evaluating is adapted from [dgl's example codes]:(https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [Espylapiza's implementation]:(https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv).

requirements: `torch, ogb, dgl, torch_geometric`

## ogbn-arxiv

Step0. `cd ogbn-arxiv`.

Step1. To reproduce the results of `AGDN+BoT`:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --seed 0 --n-label-iters 1 --lr 0.002 --model gat-ha --mode teacher --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.3 --attn_drop 0. --no-attn-dst --norm sym --n-epochs 2000 --n-runs 10 --use-labels --checkpoint-path ../checkpoint/
```

Step2. To reproduce the results of `AGDN+BoT+C&S` based on the results from Step 1:
```
CUDA_VISIBLE_DEVICES=0 python src/correct_and_smooth.py --use-norm --pred-files './checkpoint1/gat-ha/*.pt' --alpha 0.73 --n-prop 8
```

Step3. To reproduce the results of `AGDN+BoT+self-KD` based on the results from Step 1:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --seed 0 --alpha 0.9 --temp 0.7 --n-label-iters 1 --lr 0.002 --model gat-ha --mode student --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.3 --attn_drop 0. --no-attn-dst --norm sym --n-epochs 2000 --n-runs 10 --use-labels --checkpoint-path ../checkpoint/ --save-pred --pred-path ../output/
```

Step4. To reproduce the results of `AGDN+BoT+self-KD+C&S` based on the results from Step 3:
```
CUDA_VISIBLE_DEVICES=0 python src/correct_and_smooth.py --use-norm --pred-files './output/*.pt' --alpha 0.73 --n-prop 8
```

In GAT-HA, the argument "norm" means the way of extra normalization performed on "already" normalized GAT transition matrix. "norm==sym" follows the form from Espylapiza's implementation. "norm==avg" uses a mean of GAT transition matrix and GCN transition matrix.

|  model   | test_acc  |
|  ----  | ----  |
| AGDN+BoT  | 74.10 ± 0.15 |
| AGDN+BoT+C&S  | 74.16 ± 0.15 |
| AGDN+BoT+self-KD | 74.28 ± 0.17 |
| AGDN+BoT+self-KD+C&S | 74.31 ± 0.14 |


Results of previous version (you can check details in 134744923601de020dbfa51de0988d07bfbc218a):
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
