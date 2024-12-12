from models import MPNN, AGDN
import torch.nn.functional as F

def gen_model(in_feats, n_classes, args):
    use_attn_dst = not args.no_attn_dst
    residual = not args.no_residual
    bias_last = not args.no_bias_last
    in_feats_ = in_feats

    # if args.model == "gcn":
    #     model = GCN(
    #         in_feats_, 
    #         args.n_hidden, 
    #         n_classes, 
    #         args.n_layers, 
    #         F.relu, 
    #         args.dropout, 
    #         residual)

    if args.model == "mpnn":
        model = MPNN(
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
            transition_matrix=args.transition_matrix,
            residual=residual,
            bias_last=bias_last,
            no_bias=args.no_bias,
        )

    if args.model == "agdn":
        model = AGDN(
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
                diffusion_drop=args.diffusion_drop,
                use_attn_dst=use_attn_dst,
                position_emb=not args.no_position_emb,
                transition_matrix=args.transition_matrix,
                weight_style=args.weight_style,
                HA_activation=args.HA_activation,
                residual=residual,
                bias_last=bias_last,
                no_bias=args.no_bias,
                zero_inits=args.zero_inits,
                )

    # print(model)
    return model

