import math
import os
import random

import dgl
from scipy import sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

epsilon = 1 - math.log(2)

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    return torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float().to(g.device)

    
def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_inv = torch.pow(degs, -1)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_inv, deg_sqrt, deg_isqrt

def save_checkpoint(pred, n_running, checkpoint_path):
    fname = os.path.join(checkpoint_path, f'best_pred_run{n_running}.pt')
    print('Saving prediction.......')
    torch.save(pred.cpu(),fname)

def cross_entropy(x, labels):
    return F.cross_entropy(x, labels[:, 0], reduction="mean")

def loge_cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

def consis_loss(ps, temp, lam, conf=0.):
    """
    Consistency loss from GRAND [https://arxiv.org/pdf/2005.11079.pdf].
    """
    avg_p = torch.mean(ps, dim = 2)
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2)[avg_p.max(1)[0] > conf], dim = 1, keepdim=True))

    loss = lam * loss
    return loss
    
def loss_kd_only(all_out,teacher_all_out,temperature):
    T = temperature
    D_KL = torch.nn.KLDivLoss()(F.log_softmax(all_out/T, dim=1), F.softmax(teacher_all_out/T, dim=1)) * (T * T)
    return D_KL

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def add_labels(feat, labels, idx, n_classes, device):
    onehot = torch.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def plot(accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses, n_running, n_epochs):
    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, n_epochs, 100))
    ax.set_yticks(np.linspace(0, 1.0, 101))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
        plt.plot(range(n_epochs), y, label=label)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_acc_{n_running}.png")

    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, n_epochs, 100))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip(
        [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
    ):
        plt.plot(range(n_epochs), y, label=label)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_loss_{n_running}.png")

def print_info(s, verbose=1):
    if verbose:
        print(s)