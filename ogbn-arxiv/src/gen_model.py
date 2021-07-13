from models import GATHA, GAT, GCNHA, GCN
import torch.nn.functional as F

def gen_model(in_feats, n_classes, args):
    norm = args.norm
    use_attn_dst = not args.no_attn_dst
    if args.use_labels:
        in_feats_ = in_feats + n_classes
    else:
        in_feats_ = in_feats

    if args.model == "gcn":
        model = GCN(
            in_feats_, 
            args.n_hidden, 
            n_classes, 
            args.n_layers, 
            F.relu, 
            args.dropout, 
            args.use_linear)

    if args.model == "gcn-ha":
        model = GCNHA(
            in_feats_,
            n_classes,
            K=args.K,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
        )

    if args.model == "gat":
        model = GAT(
            in_feats_,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=use_attn_dst,
            use_symmetric_norm=(norm=="sym"),
        )

    if args.model == "gat-ha":
        model = GATHA(
                in_feats_,
                n_classes,
                K=args.K,
                n_hidden=args.n_hidden,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                activation=F.relu,
                dropout=args.dropout,
                input_drop=args.input_drop,
                edge_drop=args.edge_drop,
                attn_drop=args.attn_drop,
                use_attn_dst=use_attn_dst,
                norm=norm,
                )
    
    return model

