# Adaptive_Graph_Diffusion_Networks_with_Hop-wise_Attention

The preprint paper is here: https://arxiv.org/abs/2012.15024

The framework of training and evaluating is adapted from [DGL examples](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv).

requirements: `torch, ogb, dgl, torch_geometric`

## methods
We use a multi-hop AGDN layer to replace GAT layer, which incorporates multi-hop information via adaptive hop-wise attention mechanism. We further add position embedding to enhance hop information. We explicitly use these important tricks: [BoT](https://github.com/Espylapiza/Bag-of-Tricks-for-Node-Classification-with-Graph-Neural-Networks), [self-KD](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [C&S](https://github.com/Chillee/CorrectAndSmooth). AGDN is previously named as GAT-HA.

## ogbn-arxiv

Step0. `cd ogbn-arxiv`.

Step1. To reproduce the results of `AGDN`:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --seed 0 --n-label-iters 0 --standard-loss --lr 0.002 --model gat-ha --mode test --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.1 --edge_drop 0.1 --attn_drop 0. --norm none --n-epochs 2000 --n-runs 10
```

Step2. To reproduce the results of `AGDN+BoT`:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --seed 0 --n-label-iters 1 --lr 0.002 --model gat-ha --mode teacher --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.3 --attn_drop 0. --no-attn-dst --norm sym --n-epochs 2000 --n-runs 10 --use-labels --checkpoint-path ../checkpoint/
```

Step3. To reproduce the results of `AGDN+BoT+C&S` based on the results from Step 1:
```
CUDA_VISIBLE_DEVICES=0 python src/correct_and_smooth.py --use-norm --pred-files './checkpoint1/gat-ha/*.pt' --alpha 0.73 --n-prop 8
```

Step4. To reproduce the results of `AGDN+BoT+self-KD` based on the results from Step 1:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --seed 0 --alpha 0.9 --temp 0.7 --n-label-iters 1 --lr 0.002 --model gat-ha --mode student --n-layers 3 --n-hidden 256 --K 3 --n-heads 3 --dropout 0.75 --input_drop 0.25 --edge_drop 0.3 --attn_drop 0. --no-attn-dst --norm sym --n-epochs 2000 --n-runs 10 --use-labels --checkpoint-path ../checkpoint/ --save-pred --pred-path ../output/
```

Step5. To reproduce the results of `AGDN+BoT+self-KD+C&S` based on the results from Step 3:
```
CUDA_VISIBLE_DEVICES=0 python src/correct_and_smooth.py --use-norm --pred-files './output/*.pt' --alpha 0.73 --n-prop 8
```

In AGDN (or GAT-HA), the argument "norm" means the way of extra normalization performed on "already" normalized GAT transition matrix. "norm==sym" follows the form from BoT. "norm==avg" uses a mean of GAT transition matrix and GCN transition matrix.

|  model   | test_acc  |
|  ----  | ----  |
| AGDN | 73.46 ± 0.17 |
| AGDN+BoT  | 74.10 ± 0.15 |
| AGDN+BoT+C&S  | 74.16 ± 0.15 |
| AGDN+BoT+self-KD | 74.28 ± 0.17 |
| AGDN+BoT+self-KD+C&S | 74.31 ± 0.14 |


### Results of previous version

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
