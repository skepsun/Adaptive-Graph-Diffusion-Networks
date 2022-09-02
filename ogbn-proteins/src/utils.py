import random

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn.functional as F
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from gen_model import gen_model


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt

def add_labels(graph, idx, n_classes, device):
    feat = graph.srcdata["feat"]
    train_labels_onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels_onehot[idx] = graph.srcdata["train_labels_onehot"][idx]
    graph.srcdata["feat"] = torch.cat([feat, train_labels_onehot], dim=-1)

# def add_labels(graph, idx, n_classes, device, training=False):
#     feat = graph.srcdata["feat"]
#     train_labels_onehot = torch.zeros([feat.shape[0], 2*n_classes], device=device)
#     # train_labels_onehot = 0.5 * torch.ones([feat.shape[0], 2*n_classes], device=device)
#     # train_labels_onehot = torch.rand([feat.shape[0], n_classes], device=device)
#     # train_labels_onehot = torch.cat([train_labels_onehot, 1-train_labels_onehot], dim=1)
#     train_labels_onehot[idx] = graph.srcdata["train_labels_onehot"][idx].clone()
#     # train_labels_onehot = torch.cat([train_labels_onehot, 1-train_labels_onehot], dim=1)
#     if training:
#         mask = (torch.rand_like(n_classes) <= 0.9).repeat(2)
#         train_labels_onehot[idx, mask] = 0
#     graph.srcdata["feat"] = torch.cat([feat, train_labels_onehot], dim=-1)

def loge_BCE(x, labels):
    epsilon = 1 - math.log(2)
    y = F.binary_cross_entropy_with_logits(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

def plot_stats(args, train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses, n_running):
    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, args.n_epochs, 100))
    ax.set_yticks(np.linspace(0, 1.0, 101))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip([train_scores, val_scores, test_scores], ["train score", "val score", "test score"]):
        plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_score_{n_running}.png")

    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, args.n_epochs, 100))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip(
        [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
    ):
        plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_loss_{n_running}.png")
