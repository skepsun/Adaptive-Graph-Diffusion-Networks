#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import sys
import time
from copy import deepcopy
from functools import partial

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import DataLoader, NeighborSampler, SAINTSampler
from torch import nn
from tqdm import tqdm

from data import load_data, preprocess
from gen_model import count_parameters, gen_model
from sampler import RandomSampler, random_partition_v2
from utils import (add_labels, loge_loss_function, plot_stats,
                   preprocess_lp_local, seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = None
eval_device = None
dataset = "ogbn-products"
n_node_feats, n_edge_feats, n_classes = 0, 0, 0

def train(args, graph, model, dataloader, _labels, _train_idx, val_idx, test_idx, criterion, optimizer, _evaluator, estimation_mode=False):
    model.train()
    model.to(device)

    loss_sum, total = 0, 0
    train_loss = val_loss = test_loss = 1e3
    train_score = val_score = test_score = 0
    if args.sample_type == "neighbor_sample":
        for input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = torch.arange(len(output_nodes), device=device)

            if args.use_lt:
                train_labels_idx = torch.arange(len(output_nodes), len(input_nodes), device=device)
                train_pred_idx = new_train_idx
                concat = not args.label_emb

                add_labels(subgraphs[0], train_labels_idx, concat=concat)
            else:
                train_pred_idx = new_train_idx

            pred = model(subgraphs)
            loss = criterion(pred[train_pred_idx], subgraphs[-1].ndata["labels"]["_N"][train_pred_idx].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count

    if args.sample_type == "random_cluster":
        # Since we randomly split **all** nodes into several clusters, we can directly calculate val/test scores
        # based on model output during training. However, these "esitmated" scores may be far smaller than full 
        # batch scores. We only use estimated val score to select the best model.
        if estimation_mode:
            preds = torch.zeros(size=(len(_train_idx)+len(val_idx)+len(test_idx), n_classes), device=eval_device)
        if args.use_lt:
            global_train_idx = np.random.permutation(_train_idx.cpu())
            global_labels_idx = global_train_idx[:int(len(global_train_idx)*args.mask_rate)]
            global_pred_idx = global_train_idx[int(len(global_train_idx)*args.mask_rate):]
        for batch_nodes, subgraph in random_partition_v2(args.train_partition_num, graph, shuffle=True):
            subgraph = subgraph.to(device)
            new_train_idx = torch.tensor(np.random.permutation(len(batch_nodes)), device=device)
            degrees = subgraph.in_degrees()
            useful_idx = torch.arange(len(degrees))[degrees > 0]
            if args.use_lt:
                train_labels_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_labels_idx)]
                train_pred_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_pred_idx)]
                # train_labels_idx = new_train_idx[:int(len(new_train_idx)*args.mask_rate)]
                # train_pred_idx = new_train_idx[int(len(new_train_idx)*args.mask_rate):]

                add_labels(subgraph, train_labels_idx, n_classes, device)
            else:
                train_pred_idx = new_train_idx

            train_pred_idx = train_pred_idx[np.isin(batch_nodes[train_pred_idx.cpu()], _train_idx.cpu())]
            # train_pred_idx = train_pred_idx[np.isin(train_pred_idx.cpu(), useful_idx.cpu())]

            pred = model(subgraph)
            if estimation_mode:
                preds[batch_nodes] = pred.to(eval_device)
            loss = criterion(pred[train_pred_idx], subgraph.ndata["labels"][train_pred_idx, 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count
            
        if estimation_mode:
            train_loss = criterion(preds[_train_idx], _labels[_train_idx, 0].to(eval_device)).item()
            val_loss = criterion(preds[val_idx], _labels[val_idx, 0].to(eval_device)).item()
            test_loss = criterion(preds[test_idx], _labels[test_idx, 0].to(eval_device)).item()
            train_score = _evaluator(preds[_train_idx], _labels[_train_idx].to(eval_device))
            val_score = _evaluator(preds[val_idx], _labels[val_idx].to(eval_device))
            test_score = _evaluator(preds[test_idx], _labels[test_idx].to(eval_device))
    
    if args.sample_type in ["saint_node", "saint_edge", "saint_rw"]:
        # Since we randomly split **all** nodes into several clusters, we can directly calculate val/test scores
        # based on model output during training. However, these "esitmated" scores may be far smaller than full 
        # batch scores. We only use estimated val score to select the best model.
        
        if args.use_lt:
            global_train_idx = np.random.permutation(_train_idx.cpu())
            global_labels_idx = global_train_idx[:int(len(global_train_idx)*args.mask_rate)]
            global_pred_idx = global_train_idx[int(len(global_train_idx)*args.mask_rate):]
        for subgraph in dataloader:
            batch_nodes = subgraph.ndata[dgl.NID]
            subgraph = subgraph.to(device)
            new_train_idx = torch.tensor(np.random.permutation(len(batch_nodes)), device=device)
            degrees = subgraph.in_degrees()
            useful_idx = torch.arange(len(degrees))[degrees > 0]
            if args.use_lt:
                train_labels_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_labels_idx)]
                train_pred_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_pred_idx)]
                # train_labels_idx = new_train_idx[:int(len(new_train_idx)*args.mask_rate)]
                # train_pred_idx = new_train_idx[int(len(new_train_idx)*args.mask_rate):]

                add_labels(subgraph, train_labels_idx, n_classes, device)
            else:
                train_pred_idx = new_train_idx

            train_pred_idx = train_pred_idx[np.isin(batch_nodes[train_pred_idx.cpu()], _train_idx.cpu())]
            # train_pred_idx = train_pred_idx[np.isin(train_pred_idx.cpu(), useful_idx.cpu())]
            pred = model(subgraph)
            loss = criterion(pred[train_pred_idx], subgraph.ndata["labels"][train_pred_idx, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count
            
        # torch.cuda.empty_cache()

    return loss_sum / total, train_score, val_score, test_score, train_loss, val_loss, test_loss


@torch.no_grad()
def evaluate(args, graph, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator, final=False):
    model.eval()
    eval_device_ = eval_device if not final else 'cpu'
    model.to(eval_device_)

    preds = torch.zeros(labels.shape[0], n_classes, device=eval_device_)

    # Due to the memory capacity constraints, we use sampling for inference and calculate the average of the predictions 'eval_times' times.
    eval_times = args.eval_times

    for _ in range(eval_times):

        if args.sample_type == "neighbor_sample":
            for input_nodes, output_nodes, subgraphs in dataloader:
                subgraphs = [b.to(eval_device_) for b in subgraphs]
                new_train_idx = list(range(len(input_nodes)))

                if args.use_lt:
                    add_labels(subgraphs[0], new_train_idx, n_classes, eval_device_)

                pred = model(subgraphs)
                preds[output_nodes] += pred

        if args.sample_type in ["random_cluster", "saint_node", "saint_edge", "saint_rw"]:
            eval_partition_num = args.eval_partition_num if not final else 1
            eval_partition_size = graph.number_of_nodes() // eval_partition_num
            dataloader = DataLoader(graph, 
                torch.arange(graph.number_of_nodes()), 
                RandomSampler(), 
                shuffle=False, 
                num_workers=min(eval_partition_num, 4),
                batch_size=eval_partition_size)
            for batch_nodes, subgraph in random_partition_v2(args.eval_partition_num, graph, shuffle=False):
            # for batch_nodes, subgraph in dataloader:
                subgraph = subgraph.to(eval_device_)
                new_train_idx = list(range(len(batch_nodes)))
                if args.use_lt:
                    add_labels(subgraph, new_train_idx, n_classes, eval_device_)

                pred = model(subgraph)
                preds[batch_nodes] += pred


    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx, 0].to(eval_device_)).item()
    val_loss = criterion(preds[val_idx], labels[val_idx, 0].to(eval_device_)).item()
    test_loss = criterion(preds[test_idx], labels[test_idx, 0].to(eval_device_)).item()

    return (
        evaluator(preds[train_idx], labels[train_idx].to(eval_device_)),
        evaluator(preds[val_idx], labels[val_idx].to(eval_device_)),
        evaluator(preds[test_idx], labels[test_idx].to(eval_device_)),
        train_loss,
        val_loss,
        test_loss,
        preds,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]
    
    if args.sample_type == "neighbor_sample":
        train_batch_size = (len(train_idx) + args.train_partition_num - 1) // args.train_partition_num
        # train_batch_size = 100
        # batch_size = len(train_idx)
        train_sampler = NeighborSampler([32] * args.n_layers)
        train_dataloader = DataLoader(
            graph.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=10,
        )

        eval_batch_size = (len(train_idx) + args.eval_partition_num - 1) // args.eval_partition_num
        eval_sampler = NeighborSampler([100 for _ in range(args.n_layers)])
        # sampler = MultiLayerFullNeighborSampler(args.n_layers)
        eval_dataloader = DataLoader(
            graph.cpu(),
            torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            eval_sampler,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )
    
    if args.sample_type == "random_cluster":
        train_dataloader = None
        eval_dataloader = None
        test_idx_during_training = test_idx
        # train_dataloader = RandomPartition(args.train_partition_num, graph, shuffle=True)
        # eval_dataloader = RandomPartition(args.eval_partition_num, graph, shuffle=False)

    
    if args.sample_type in ["saint_node", "saint_edge" ,"saint_rw"]:
        test_idx_during_training = test_idx
        if args.sample_type == 'saint_node':
            mode = 'node'
            budget = args.node_budget
        if args.sample_type == 'saint_edge':
            mode = 'edge'
            budget = args.edge_budget
        if args.sample_type == 'saint_rw':
            mode = 'walk'
            budget = args.rw_budget
        
        train_sampler = SAINTSampler(mode, budget)
        train_dataloader = DataLoader(
            graph, torch.arange(args.n_subgraphs), train_sampler,
            batch_size=args.saint_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=8)
        eval_batch_size = (len(labels) + args.eval_partition_num - 1) // args.eval_partition_num
        eval_dataloader = None

    if args.loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "loge":
        criterion = loge_loss_function


    model = gen_model(args, n_node_feats, n_edge_feats, n_classes).to(device)

    if args.advanced_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.7, patience=20, verbose=True, min_lr=1e-4
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None

    for epoch in range(1, args.n_epochs + 1):
        log_flag = epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0
        tic = time.time()
        loss, train_score, val_score, test_score, train_loss, val_loss, test_loss \
            = train(args, graph, model, train_dataloader, labels, train_idx, val_idx, test_idx, criterion, optimizer, evaluator_wrapper, estimation_mode=(args.estimation_mode) and log_flag)

        toc = time.time()
        total_time += toc - tic

        if log_flag:
            if not (args.estimation_mode and args.sample_type in ["random_cluster"]):
                train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                    args, graph, model, eval_dataloader, labels, train_idx, val_idx, test_idx_during_training, criterion, evaluator_wrapper
                )
            
            eval_time = time.time() - toc

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                # final_pred = pred
                best_model = deepcopy(model)

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s, Evaluate time: {eval_time:.2f}s"
                )
                print(
                    f"Loss: {loss:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}"
                )

            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

        if args.advanced_optimizer:
            lr_scheduler.step(val_score)

    tic = time.time()
    final_train_score, best_val_score, final_test_score, _, _, _, final_pred = evaluate(
                args, graph, best_model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper, final=True
            )
    toc = time.time()
    print("*" * 50)
    print(f"Best val score: {best_val_score}, Final test score: {final_test_score}, Full evaluation time: {(toc-tic):.4f}s")
    print("*" * 50)

    if args.plot:
        plot_stats(args, train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses, n_running)

    if args.save_pred:
        os.makedirs("../output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"../output/{n_running-1}.pt")

    return best_val_score, final_test_score


def main():
    global device, eval_device, n_node_feats, n_edge_feats, n_classes

    argparser = argparse.ArgumentParser(
        "GAT & AGDN implementation on ogbn-products", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--root", type=str, default="/mnt/ssd/ssd/dataset")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=250, help="number of epochs")
    argparser.add_argument("--advanced-optimizer", action="store_true")
    argparser.add_argument("--model", type=str, default="agdn", choices=["gat", "agdn"])
    argparser.add_argument("--sample-type", type=str, default="random_cluster", 
        choices=["neighbor_sample", "random_cluster", "khop_sample", "saint_node", "saint_edge", "saint_rw"])
    argparser.add_argument("--train-partition-num", type=int, default=10, 
        help="number of partitions for training")
    argparser.add_argument("--eval-partition-num", type=int, default=1, 
        help="number of partitions for evaluating")
    argparser.add_argument('--saint-batch-size', type=int, default=256)
    argparser.add_argument('--node-budget', type=int, default=80000)
    argparser.add_argument('--edge-budget', type=int, default=30000)
    argparser.add_argument('--rw-budget', type=int, nargs='+', default=[20000, 3])
    argparser.add_argument('--n-subgraphs', type=int, default=1000,
                        help='The subgraph number for graphsaint sampler')
    argparser.add_argument('--fanouts', type=int, nargs='+', default=[15,10,5],
                        help='The neighbor size of each hop for neighborsampler/shadowkhopsampler')

    argparser.add_argument("--loss-type", type=str, default="cross_entropy", choices=["cross_entropy", "loge"])
    argparser.add_argument("--use-lt", action="store_true", 
        help="Use labels in the training set as input features.")
    argparser.add_argument("--mask-rate", type=float, default=0.5, 
        help="rate of labeled nodes at each epoch, which only takes effect when sample_type==random_cluster & use_labels=True")

    argparser.add_argument("--eval-gpu", type=int, default=-1)
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=4, help="number of heads")
    argparser.add_argument("--norm", type=str, default="none", choices=["none", "adj", "avg"])
    argparser.add_argument(
        "--estimation-mode", action="store_true", help="Estimate the score of test set for speed during training."
    )
    argparser.add_argument("--K", type=int, default=3)
    argparser.add_argument("--eval-times", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=4, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=120, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=10, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=10, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")

    args = argparser.parse_args()
    print(args)

    print(f"Estimation mode during training: {args.estimation_mode} (Estimated val/test scores may be lower than final ones)")
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    eval_device = torch.device(f"cuda:{args.eval_gpu}") if args.eval_gpu >= 0 else torch.device("cpu")
    print(device, eval_device)
    # load data & preprocess
    print("Loading data")
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, args)
    print("Preprocessing")
    graph, labels = preprocess(graph, labels, train_idx, n_classes, args)
    n_node_feats = graph.ndata["feat"].shape[-1]
    n_classes = (labels.max() + 1).item()
    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []

    for i in range(args.n_runs):
        print("Running", i)
        seed(args.seed + i)
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args, n_node_feats, n_edge_feats, n_classes)}")


if __name__ == "__main__":
    main()
