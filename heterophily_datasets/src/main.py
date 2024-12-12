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
from dgl.data import ChameleonDataset, SquirrelDataset, ActorDataset

from gen_model import gen_model
from utils import (add_labels, adjust_learning_rate, compute_acc, compute_norm,
                   cross_entropy, loge_cross_entropy, plot, print_info, seed,
                   split_dataset, index_to_mask)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = None
in_feats, n_classes = None, None


def train(args, model, graph, labels, train_mask, optimizer, loss_fcn, epoch=1):
    model.train()

    feat = graph.ndata["feat"]
    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = loss_fcn(pred[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    return compute_acc(pred[train_mask], labels[train_mask]), loss


@torch.no_grad()
def evaluate(args, model, graph, labels, train_mask, val_mask, test_mask, loss_fcn):
    model.eval()

    feat = graph.ndata["feat"]

    pred = model(graph, feat)
    train_loss = loss_fcn(pred[train_mask], labels[train_mask])
    val_loss = loss_fcn(pred[val_mask], labels[val_mask])
    test_loss = loss_fcn(pred[test_mask], labels[test_mask])

    return (
        compute_acc(pred[train_mask], labels[train_mask]),
        compute_acc(pred[val_mask], labels[val_mask]),
        compute_acc(pred[test_mask], labels[test_mask]),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_mask, val_mask, test_mask, n_running):
    # define model and optimizer
    model = gen_model(in_feats, n_classes, args)
    model = model.to(device)
    print_info(f"Number of params: {count_parameters(args)}", verbose=args.verbose)

    if not args.standard_loss:
        loss_fcn = loge_cross_entropy
    else:
        loss_fcn = cross_entropy
    if args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    ### do nomalization only once
    
    deg_inv, deg_sqrt, deg_isqrt = compute_norm(graph)
    
    
    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_isqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_sqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

    graph.edata["sage_norm"] = deg_inv[graph.edges()[1]]

    checkpoint_path = args.checkpoint_path
    count = 0
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.adjust_lr:
            adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_mask, optimizer, loss_fcn, epoch=epoch)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_mask, val_mask, test_mask, loss_fcn
        )

        toc = time.time()
        total_time += toc - tic

        if args.selection_metric == "acc":
            new_best = val_acc > best_val_acc
        else:
            new_best = val_loss < best_val_loss
        if new_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            final_pred = pred
            count = 0
        else:
            count += 1


        if epoch % args.log_every == 0:
            print_info(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}", verbose=args.verbose)
            print_info(f"Time: {(total_time / epoch):.4f}, Loss: {loss.item():.4f}, Acc: {acc:.4f}", verbose=args.verbose)
            print_info(f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}", verbose=args.verbose)
            print_info(f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}", verbose=args.verbose)

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
        ):
            l.append(e)
        if count == args.patience:
            break

    print_info("*" * 50, verbose=args.verbose)
    print_info(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}", verbose=args.verbose)

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
    # print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("AGDN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset and device setting
    argparser.add_argument("--dataset", type=str, default="chameleon")
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help   ="GPU device ID.")
    argparser.add_argument("--root", type=str, default="/mnt/ssd/ssd/dataset")
    argparser.add_argument("--no-self-loops", action="store_true", help="Do not add self-loops.")
    argparser.add_argument("--train_size", type=float, default=0.6)
    argparser.add_argument("--val_size", type=float, default=0.2)

    # Training setting
    argparser.add_argument("--seed", type=int, default=0, help="initial random seed.")
    argparser.add_argument("--selection-metric", type=str, default="acc", choices=["acc", "loss"])
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--optimizer", type=str, default="adam", choices=["rmsprop", "adam"])
    argparser.add_argument("--wd", type=float, default=5e-4)
    argparser.add_argument("--alpha",type=float,default=0.5,help="ratio of kd loss")
    argparser.add_argument("--temp",type=float,default=1.0,help="temperature of kd")
    argparser.add_argument("--patience", type=int, default=2000)
    # BoT
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--no-residual", action="store_true", help="Don't use residual linears")
    argparser.add_argument("--adjust-lr", action="store_true", help="adjust learning rate in first 50 iterations")
    argparser.add_argument("--standard-loss", action="store_true")

    # Model setting
    argparser.add_argument("--model", type=str, default="agdn")
    argparser.add_argument("--no-bias", action="store_true")
    argparser.add_argument("--no-bias-last", action="store_true")
    argparser.add_argument("--batch-norm", action="store_true")
    argparser.add_argument("--n-layers", type=int, default=2)
    argparser.add_argument("--n-heads", type=int, default=1)
    argparser.add_argument("--n-hidden", type=int, default=512)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--input_drop", type=float, default=0.0)
    argparser.add_argument("--edge_drop", type=float, default=0.0)
    argparser.add_argument("--attn_drop", type=float, default=0.0)
    argparser.add_argument("--diffusion_drop", type=float, default=0.0)
    # AGDN setting
    argparser.add_argument("--K", type=int, default=2)
    argparser.add_argument("--no-position-emb", action="store_true")
    argparser.add_argument("--transition-matrix", type=str, default="gat_adj")
    argparser.add_argument("--weight-style", type=str, default="HA")
    argparser.add_argument("--HA-activation", type=str, default="leakyrelu")
    argparser.add_argument("--zero-inits", action="store_true")

    # Print setting
    argparser.add_argument("--verbose", type=int, default=1)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    # Saving setting
    argparser.add_argument("--checkpoint-path", type=str, default="../checkpoint/")
    argparser.add_argument("--output-path", type=str, default="../output/")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    
    args = argparser.parse_args()
    print_info(f"args: {args}", verbose=args.verbose)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    # load data
    if args.dataset == "chameleon":
        data = ChameleonDataset()
    elif args.dataset == "squirrel":
        data = SquirrelDataset()
    elif args.dataset == "actor":
        data = ActorDataset()

    
    graph = data[0]
    labels = graph.ndata["label"]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    # In DGL implementation, we remove existing self loops first then add full self loops,
    # which is different from the add_remaining_loops in PyG. PyG simply keep existing self loops and add
    # remaining ones, which cannot ensure that each node has **only one** self loop.
    if not args.no_self_loops:
        print_info(f"Total edges before adding self-loop {graph.number_of_edges()}", verbose=args.verbose)
        graph = graph.remove_self_loop().add_self_loop()
        print_info(f"Total edges after adding self-loop {graph.number_of_edges()}", verbose=args.verbose)

    # graph.ndata['PE'] = torch.load("/mnt/ssd/ssd/CorrectAndSmooth/embeddings/spectralarxiv.pt", map_location=graph.device)
    # graph.ndata['PE'] = positional_encoding(graph, 8)
    # PE = [graph.ndata['PE']]
    # for _ in range(args.K):
    #     graph.update_all(fn.copy_u('PE', 'm'), fn.mean('m', 'PE'))
    #     PE.append(graph.ndata['PE'])
    # graph.ndata['PE'] = torch.stack(PE, dim=1)
    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(1, args.n_runs + 1):
        seed(i + args.seed)
        indices = split_dataset(labels, args.train_size, args.val_size, 1 - args.train_size - args.val_size)
        train_mask, val_mask, test_mask = index_to_mask(indices[0], size=len(labels)), index_to_mask(indices[1], size=len(labels)), index_to_mask(indices[2], size=len(labels))
        print_info(f"Num training nodes: {torch.sum(train_mask)}", verbose=args.verbose)
        print_info(f"Num validation nodes: {torch.sum(val_mask)}", verbose=args.verbose)
        print_info(f"Num test nodes: {torch.sum(test_mask)}", verbose=args.verbose)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        val_acc, test_acc = run(args, graph, labels, train_mask, val_mask, test_mask, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print_info(f"Runned {args.n_runs} times", verbose=args.verbose)
    print_info(f"Val Accs: {val_accs}", verbose=args.verbose)
    print_info(f"Test Accs: {test_accs}", verbose=args.verbose)
    print(f"Number of params: {count_parameters(args)}, Average val/test accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}/{np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    


if __name__ == "__main__":
    main()



