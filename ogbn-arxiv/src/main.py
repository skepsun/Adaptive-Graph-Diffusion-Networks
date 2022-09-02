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
from utils import (add_labels, adjust_learning_rate, compute_acc, compute_norm, positional_encoding,
                   cross_entropy, loge_cross_entropy, loss_kd_only, consis_loss, plot, print_info,
                   save_checkpoint, seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = None
in_feats, n_classes = None, None


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, teacher_output, loss_fcn, evaluator, epoch=1):
    model.train()

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx, n_classes, device)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        # We change mask to ~mask to match previous definition
        train_pred_idx = train_idx[~mask]

    optimizer.zero_grad()
    if args.use_consis_loss:
        p_list = []
        loss = 0
        for s in range(args.sample):
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
            if not args.strict_consis_loss:
                p_list.append(pred.softmax(-1))
            else:
                p_list.append(pred.softmax(-1)[torch.cat([val_idx, test_idx])])
            loss += loss_fcn(pred[train_pred_idx], labels[train_pred_idx])
        loss /= args.sample
        ps = torch.stack(p_list, dim=2)
        loss_consis = consis_loss(ps, args.consis_temp, args.consis_lamb, conf=args.conf)
        loss += loss_consis
    else:
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
        
        loss = loss_fcn(pred[train_pred_idx], labels[train_pred_idx])
    if args.mode == "student":
        loss_kd = loss_kd_only(pred, teacher_output, args.temp)
        loss = loss*(1-args.alpha) + loss_kd*args.alpha
    loss.backward()
    optimizer.step()

    return compute_acc(pred[train_idx], labels[train_idx], evaluator), loss


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, use_labels, loss_fcn, evaluator):
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
    train_loss = loss_fcn(pred[train_idx], labels[train_idx])
    val_loss = loss_fcn(pred[val_idx], labels[val_idx])
    test_loss = loss_fcn(pred[test_idx], labels[test_idx])

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
    if args.mode == "student":
        teacher_output = torch.load(os.path.join(checkpoint_path, f'best_pred_run{n_running}.pt')).cpu().cuda()
    else:
        teacher_output = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.adjust_lr:
            adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, teacher_output, loss_fcn, evaluator, epoch=epoch)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, loss_fcn, evaluator
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
            if args.mode == "teacher":
                os.makedirs(checkpoint_path, exist_ok=True)
                save_checkpoint(final_pred, n_running, checkpoint_path)

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
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--root", type=str, default="/mnt/ssd/ssd/dataset")
    argparser.add_argument("--no-self-loops", action="store_true", help="Do not add self-loops.")
    argparser.add_argument("--use-xrt-emb", action="store_true")

    # Training setting
    argparser.add_argument("--seed", type=int, default=0, help="initial random seed.")
    argparser.add_argument("--mode", type=str, default="test", choices=["test", "student", "teacher"])
    argparser.add_argument("--selection-metric", type=str, default="loss", choices=["acc", "loss"])
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adam"])
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--alpha",type=float,default=0.5,help="ratio of kd loss")
    argparser.add_argument("--temp",type=float,default=1.0,help="temperature of kd")
    
    # BoT
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--no-residual", action="store_true", help="Don't use residual linears")
    argparser.add_argument("--adjust-lr", action="store_true", help="adjust learning rate in first 50 iterations")
    argparser.add_argument("--standard-loss", action="store_true")

    # Consistence loss
    argparser.add_argument('--use-consis-loss', action='store_true',)
    argparser.add_argument('--strict-consis-loss', action='store_true',)
    argparser.add_argument('--consis-temp', type=float, default=0.5,)
    argparser.add_argument('--consis-lamb', type=float, default=1.,)
    argparser.add_argument('--sample', type=int, default=1,)
    argparser.add_argument('--conf', type=float, default=0.,)

    # Model setting
    argparser.add_argument("--model", type=str, default="agdn")
    argparser.add_argument("--no-bias", action="store_true")
    argparser.add_argument("--no-bias-last", action="store_true")
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-heads", type=int, default=1)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--input_drop", type=float, default=0.0)
    argparser.add_argument("--edge_drop", type=float, default=0.0)
    argparser.add_argument("--attn_drop", type=float, default=0.0)
    argparser.add_argument("--diffusion_drop", type=float, default=0.0)
    # AGDN setting
    argparser.add_argument("--K", type=int, default=3)
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
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=args.root)
    
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    print_info(f"Num training nodes: {len(train_idx)}", verbose=args.verbose)
    print_info(f"Num validation nodes: {len(val_idx)}", verbose=args.verbose)
    print_info(f"Num test nodes: {len(test_idx)}", verbose=args.verbose)
    graph, labels = data[0]
    if args.use_xrt_emb:
        print_info(f"raw node feature size: {graph.ndata['feat'].shape[-1]}", verbose=args.verbose)
        graph.ndata["feat"] = torch.from_numpy(np.load("/home/scx/dataset/ogbn_arxiv/processed/X.all.xrt-emb.npy")).float()
        print_info(f"new node feature size: {graph.ndata['feat'].shape[-1]}", verbose=args.verbose)

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

    print_info(f"Runned {args.n_runs} times", verbose=args.verbose)
    print_info(f"Val Accs: {val_accs}", verbose=args.verbose)
    print_info(f"Test Accs: {test_accs}", verbose=args.verbose)
    print(f"Number of params: {count_parameters(args)}, Average val/test accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}/{np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    


if __name__ == "__main__":
    main()



