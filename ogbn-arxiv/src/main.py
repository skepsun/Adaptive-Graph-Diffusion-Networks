#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time

import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gen_model import gen_model
from utils import (add_labels, adjust_learning_rate, compute_acc, compute_norm,
                   loge_cross_entropy, loss_kd_only, plot, save_checkpoint, seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = None
in_feats, n_classes = None, None


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, teacher_output, evaluator, epoch=1):
    model.train()

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx, n_classes, device)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    if args.n_label_iters > 0 and args.use_labels:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            # unlabel_probs = F.softmax(pred[unlabel_idx], dim=-1)
            # unlabel_preds = torch.argmax(unlabel_probs, dim=-1)
            # confident_unlabel_idx = unlabel_idx[unlabel_probs.max(dim=-1)[0] > 0.7]
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)
    loss = loge_cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    if args.mode == "student":
        loss_kd = loss_kd_only(pred, teacher_output, args.temp)
        loss = loss*(1-args.alpha) + loss_kd*args.alpha
    loss.backward()
    optimizer.step()

    return compute_acc(pred[train_idx], labels[train_idx], evaluator), loss


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx, n_classes, device)

    pred = model(graph, feat)
    if args.n_label_iters > 0 and args.use_labels:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)
    train_loss = loge_cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = loge_cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = loge_cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    # define model and optimizer
    model = gen_model(in_feats, n_classes, args)
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    ### do nomalization for only one time
    
    deg_sqrt, deg_isqrt = compute_norm(graph)
    
    
    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_isqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_sqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

    checkpoint_path = os.path.join(args.checkpoint_path, args.model)
    if args.mode == "student":
        teacher_output = torch.load(os.path.join(checkpoint_path, f'best_pred_run{n_running}.pt')).cpu().cuda()
    else:
        teacher_output = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.adjust_lr:
            adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, teacher_output, evaluator, epoch=epoch)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            final_pred = pred
            if args.mode == "teacher":
                os.makedirs(checkpoint_path, exist_ok=True)
                save_checkpoint(final_pred, n_running, checkpoint_path)

        if epoch % args.log_every == 0:
            print(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}", )
            print(f"Time: {(total_time / epoch):.4f}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
            print(f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}")
            print(f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}")

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")

    if args.plot_curves:
        plot(accs, train_accs, val_accs, test_accs, 
             losses, train_losses, val_losses, test_losses, 
             n_running, args.n_epochs)

    if args.save_pred:
        os.makedirs(args.output_path, exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), os.path.join(args.output_path, f"{n_running - 1}.pt"))

    return best_val_acc, best_test_acc


def count_parameters(args):
    model = gen_model(in_feats, n_classes, args)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("AGDN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--root", type=str, default="../dataset")
    argparser.add_argument("--model", type=str, default="gat-ha")
    argparser.add_argument("--seed", type=int, default=0, help="initial random seed.")
    argparser.add_argument("--mode", type=str, default="test")
    argparser.add_argument("--alpha",type=float,default=0.5,help="ratio of kd loss")
    argparser.add_argument("--temp",type=float,default=1.0,help="temperature of kd")
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--norm", type=str, help="Choices of normalization methods. values=['none','sym','avg']", default='none')
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--K", type=int, default=3)
    argparser.add_argument("--n-heads", type=int, default=1)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--input_drop", type=float, default=0.0)
    argparser.add_argument("--edge_drop", type=float, default=0.0)
    argparser.add_argument("--attn_drop", type=float, default=0.05)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--use-linear", action="store_true", help="only useful for gcn model")
    argparser.add_argument("--checkpoint-path", type=str, default="../checkpoint/")
    argparser.add_argument("--output-path", type=str, default="../output/")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--adjust-lr", action="store_true", help="adjust learning rate in first 50 iterations")
    args = argparser.parse_args()
    print(f"args: {args}")
    assert args.mode in ["teacher", "student", "test"]

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=args.root)
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(1, args.n_runs + 1):
        seed(i + args.seed)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Runned {args.n_runs} times")
    print(f"Val Accs: {val_accs}")
    print(f"Test Accs: {test_accs}")
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")
    


if __name__ == "__main__":
    main()



