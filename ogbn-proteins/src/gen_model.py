import numpy as np
import torch.nn.functional as F

from models import GAT, AGDN


def gen_model(args, n_node_feats, n_edge_feats, n_classes):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.model == "gat":
        model = GAT(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=16,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot_feature=args.use_one_hot_feature,
            use_labels=args.use_labels,
        )

    if args.model == "agdn":
        model = AGDN(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=16,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            hop_attn_drop=args.hop_attn_drop,
            edge_drop=args.edge_drop,
            K=args.K,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            use_labels=args.use_labels,
            weight_style=args.weight_style,
        )

    return model


def count_parameters(args, n_node_feats, n_edge_feats, n_classes):
    model = gen_model(args, n_node_feats, n_edge_feats, n_classes)
    n_parameters = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    del model
    return n_parameters
