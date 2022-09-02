# Adaptive Graph Diffusion Networks

This is a pytorch implementation of the paper [Adaptive Graph Diffusion Networks](https://arxiv.org/abs/2012.15024).

## Environment
We conduct all experiments on a **single Tesla V100 (16Gb) card**. The maximum memory cost on certain datasets may be up to ~32Gb. We implement our methods with Deep Graph Library (DGL). Check `requirements.txt` for more associated python packages.

## Reproduce
We offer associated shell scripts for reproducing our results. Remember to modify your own dataset root path in each directory. We also offer other common models and possible sampling methods in some directories (not fully evaluated).

Feel free to utilize and modify this repository, but remember to briefly introduce and cite this work:D
 
## Method
We propose Adaptive Graph Diffusion Networks (AGDNs) to extend receptive fields of common Message Passing Nueral Networks (MPNNs), without extra layers (feature transformations) or decoupling model architecture (resulting graph convolution restricted in the same space). Following the historical path of GNNs, that spectral GNNs have elegant analyzability but usually have poor scalability and performance, we generalize the graph diffusion to be more spatial. In detail, for an MPNN model (usually GAT in this repository), we replace the graph convolution operator in each layer with a generalized graph diffusion operator. The generalized graph diffusion operator is defined as follows:

$$\tilde{\boldsymbol H}^{(l,0)} = \boldsymbol H^{(l-1)}\boldsymbol W^{(l)},$$

$$\tilde{\boldsymbol H}^{(l,k)} = \overline{\boldsymbol A}\tilde{\boldsymbol H}^{(l,k-1)},$$

$${\boldsymbol H}^{(l)}=\sum^{K}_{k=0}{\boldsymbol \Theta}^{(k)}\otimes\tilde{\boldsymbol H}^{(l,k)}+\boldsymbol{H}^{(l-1)}\boldsymbol{W}^{(l),r},$$

where $\otimes$ denotes the element-wise matrix multiplication. We describe the above proceedures in a node viewpoint:

$$\tilde{\boldsymbol h}^{(l,0)}_i=\boldsymbol h^{(l-1)}_i\boldsymbol W^{(l)},$$

$$\tilde{\boldsymbol h}^{(l,k)}_i=\sum_{j\in \mathcal{N}_i}\overline{A}_{ij}\tilde{\boldsymbol h}^{(l,k-1)}_j,$$

$$h^{(l)}_{ic}=\sum_{k=0}^{K}\theta_{ikc}\tilde{h}^{(l,k)}_{ic}+\sum_{c'=1}^{d^{(l-1)}}h^{(l-1)}_{ic'}W^{(l),r}_{c'c},$$

To obtain the possibly node-wise or channel-wise weighting coefficients $\theta_{ikc}$, we propose two mechanisms: Hop-wise Attention (HA) and Hop-wise Convolution (HC).

Hop-wise Attention (HA) is a GAT-like attention mechanism and utilizes a $2\times d$ query vector $\boldsymbol a_{hw}$ to induce node-wise weighting coefficients $\boldsymbol \Theta^{HA} \in \mathbb R^{N\times (K+1)}$:

$$\omega_{ik} = \left[\tilde{\boldsymbol h}^{(l,0)}_{i}\left\lvert\right\rvert\tilde{\boldsymbol h}^{(l,k)}_{i}\right]\cdot \boldsymbol a_{hw},$$

$${\theta}_{ik}^{HA}=\frac {{\rm exp}\left({\sigma}\left(\omega_{ik}\right)\right)}{\sum _{k=0}^{K} {{\rm exp}\left({\sigma}\left(\omega_{ik}\right)\right)}},$$

$$\theta^{HA}_{ikc}=\theta^{HA}_{ik},\forall i,k,c$$

Hop-wise Convolution (HC) directly defines a learnable channel-wise convolution kernel $\boldsymbol \Theta^{HC}\in \mathbb R^{(K+1)\times d}$:

$$\theta^{HC}_{ikc}=\theta^{HC}_{kc},\forall i,k,c$$

HA and HC are just examples, we expect more mechanisms from the community.

In addition, we utilize hop-wise Positional Embedding (PE) on some datasets to enhance hop-wise position information (PE may increase 1~2 Gb memory cost):

$$\tilde{\boldsymbol h}^{(l,k)} = \tilde{\boldsymbol h}^{(l,k)} + \boldsymbol p^{(l,k)}$$

## Performance
ogbn-arxiv:
|  Model   | Test Accuracy (%) | Validation Accuracy (%) |
|  ----  | ----  | ----  |
| GCN | 71.74±0.29 | 73.00±0.17 |
| GraphSAGE | 71.49±0.27 | 72.77±0.16 |
| DeeperGCN | 71.92±0.16 | 72.62±0.14 |
| JKNet | 72.19±0.21 | 73.35±0.07 |
| DAGNN | 72.09±0.25 | 72.90±0.11 |
| GCNII | 72.74±0.16 | – |
| MAGNA | 72.76±0.14 | – |
| UniMP | 73.11±0.20 | 74.50±0.15 |
| GAT+BoT | 73.910.12 | 75.16±0.08 |
| RevGAT+BoT | 74.02±0.18 | 75.01±0.10 |
| AGDN+BoT | 74.11±0.12 | 75.25±0.05 |
| GAT+BoT+self-KD | 74.16±0.08 | 75.14±0.04 |
| RevGAT+BoT+self-KD | 74.26±0.17 | 74.97±0.08 |
| AGDN+BoT+self-KD | 74.31±0.12 | 75.22±0.09 |
| RevGAT+XRT+BoT | 75.90±0.19 | 77.01±0.09 |
| AGDN+XRT+BoT | 76.18±0.16 | 77.24±0.06 |
| RevGAT+XRT+BoT+self-KD | 76.15±0.10 | 77.16±0.09 |
| AGDN+XRT+BoT+self-KD | 76.37±0.11 | 77.19±0.08 |

ogbn-proteins:
|  Model   | Test ROC-AUC (%) | Validation ROC-AUC (%) |
|  ----  | ----  | ----  |
| GCN  | 72.51±0.35 | 79.21±0.18 |
| GraphSAGE | 77.68±0.20 | 83.34±0.13 |
| DeeperGCN | 85.80±0.17 | 91.06±0.16 |
| UniMP | 86.42±0.08 | 91.75±0.06 |
| GAT+BoT | 87.65±0.08 | 92.80±0.08 |
| RevGNN-deep | 87.74±0.13 | 93.26±0.06 |
| RevGNN-wide | 88.24±0.15 |94.50±0.08 |
| AGDN |88.65±0.13 | 94.18±0.05 |

ogbn-products:
|  Model   | Test Accuracy (%) | Validation Accuracy (%) |
|  ----  | ----  | ----  |
| GCN | 75.64±0.21 | 92.00±0.03 |
| GraphSAGE | 78.50±0.14 | 92.24±0.07 |
| GraphSAINT | 80.27±0.26 | – |
| DeeperGCN | 80.98±0.20 | 92.38±0.09 |
| SIGN | 80.52±0.16 | 92.99±0.04 |
| UniMP | 82.56±0.31 | 93.08±0.17 |
| RevGNN-112 | 83.07±0.30 | 92.90±0.07 |
| AGDN | 83.34±0.27 | 92.29±0.10 |

ogbl-ppa:
|  Model   | Test Hits@100 (%) | Validation Hits@100 (%) |
|  ----  | ----  | ----  |
| DeepWalk | 28.88±1.53 | - |
| Matrix Factorization | 32.29±0.94 | 32.28±4.28 |
| Common Neighbor | 27.65±0.00 | 28.23±0.00 |
| Adamic Adar | 32.45±0.00 | 32.68±0.00 |
| Resource Allocation | 49.33±0.00 | 47.22±0.00 |
| GCN | 18.67±1.32 | 18.45±1.40 |
| GraphSAGE | 16.55±2.40 | 17.24±2.64 |
| SEAL | 48.80±3.16 | 51.25±2.52 |
| PLNLP | 32.38±2.58 | - |
| Ours (AGDN) | 41.23±1.59 | 43.32±0.92 |

ogbl-ddi:
|  Model   | Test Hits@20 (%) | Validation Hits@20 (%) |
|  ----  | ----  | ----  |
| DeepWalk | 22.46±2.90 | – |
| Matrix Factorization | 13.68±4.75 | 33.70±2.64 |
| Common Neighbor | 17.73±0.00 | 9.47±0.00 |
| Adamic Adar | 18.61±0.00 | 9.66±0.00 |
| Resource Allocation | 6.23±0.00 | 7.25±0.00 |
| GCN | 37.07±5.07 | 55.50±2.08 |
| GraphSAGE | 53.90±4.74 | 62.62±0.37 |
| SEAL | 30.56±3.86 | 28.49±2.69 |
| PLNLP | 90.88±3.13 | 82.42±2.53 |
| Ours (AGDN) | 95.38±0.94 | 89.43±2.81 |

ogbl-citation2:
|  Model   | Test MRR (%) | Validation MRR (%) |
|  ----  | ----  | ----  |
| Matrix Factorization | 51.86±4.43 | 51.81±4.36 |
| Common Neighbor | 51.47±0.00 | 51.19±0.00 |
| Adamic Adar | 51.89±0.00 | 51.67±0.00 |
| Resource Allocation | 51.98±0.00 | 51.77±0.00 |
| GCN  | 84.74±0.31 | 84.79±0.23 |
| GraphSAGE | 82.60±0.36 | 82.63±0.33 |
| SEAL | 87.67±0.32 | 87.57±0.31 |
| PLNLP | 84.92±0.29 | 84.90±0.31 |
| Ours (AGDN) | 85.49±0.29 | 85.56±0.33 |

## Performance
ogbn-proteins (The inference runtime on another RTX 6000 (48Gb) card of RevGNN is not reported in its paper):
| Model | Training Runtime | Inference Runtime | Parameters |
| ----- | ---- | ---- | ---- |
| RevGNN-Deep | 13.5d/2000epochs | – | 20.03M |
| RevGNN-Wide | 17.1d/2000epochs | – | 68.47M |
| AGDN | 0.14d/2000epochs | 12s | 8.61M |

ogbl-ppa, ogbl-ddi, ogbl-citation2:
| Dataset | Model | Training Runtime | Inference Runtime | Parameters |
| ---- | ---- | ---- | ---- | ---- |
| ogbl-ppa | SEAL | 20h/20epochs | 4h | 0.71M |
| ogbl-ppa | AGDN | 2.3h/40epochs | 0.06h | 36.90M |
| ogbl-ddi | SEAL | 0.07h/10epochs | 0.1h | 0.53M |
| ogbl-ddi | AGDN | 0.8h/2000epochs | 0.3s | 3.51M |
| ogbl-citation2 | SEAL | 7h/10epochs | 28h | 0.26M |
| ogbl-citation2 | AGDN | 2.5h/2000epochs | 0.06h | 0.31M |

## Extra tricks

For ogbn-arxiv: [BoT](https://github.com/Espylapiza/Bag-of-Tricks-for-Node-Classification-with-Graph-Neural-Networks), [self-KD](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [GIANT-XRT](https://github.com/elichienxD/deep_gcns_torch).

For ogbn-ddi: AUC loss from [PLNLP](https://github.com/zhitao-wang/PLNLP).

## Reference
1. BoT ([Repository](https://github.com/Espylapiza/Bag-of-Tricks-for-Node-Classification-with-Graph-Neural-Networks), [Paper](https://arxiv.org/abs/2103.13355))
2. Self-KD([Repository](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv))
3. GIANT-XRT([Repository](https://github.com/elichienxD/deep_gcns_torch), [Paper](https://arxiv.org/abs/2111.00064))
4. PLNLP([Repository](https://github.com/zhitao-wang/PLNLP), [Paper](https://arxiv.org/pdf/2112.02936))
5. GraphSAINT([Repository](https://github.com/GraphSAINT/GraphSAINT), [Paper](https://openreview.net/forum?id=BJe8pkHFwS))
6. DeeperGCN ([Repository](https://github.com/lightaime/deep_gcns_torch), [Paper](https://arxiv.org/abs/2006.07739))
7. RevGNNs ([Repository](https://github.com/lightaime/deep_gcns_torch/tree/master/examples/ogb_eff), [Paper](https://github.com/lightaime/deep_gcns_torch))
8. UniMP ([Repository](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp), [Paper](https://arxiv.org/pdf/2009.03509))
9. SEAL ([Repository](https://github.com/facebookresearch/SEAL_OGB), [Paper](https://arxiv.org/pdf/2010.16103))