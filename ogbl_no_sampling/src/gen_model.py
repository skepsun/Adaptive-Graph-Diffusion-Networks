from models import GCN, GAT, SAGE, AGDN, MemAGDN, DotPredictor, CosPredictor, LinkPredictor

def gen_model(args, in_feats, in_edge_feats, device):
    if args.model == 'gcn':
        model = GCN(in_feats, args.n_hidden,
                    args.n_hidden, args.n_layers,
                    args.dropout, args.input_drop,
                    bn=args.bn, residual=args.residual).to(device)
    if args.model == 'gat':
        model = GAT(in_feats, args.n_hidden,
                    args.n_hidden, args.n_layers,
                    args.n_heads,
                    args.dropout, args.input_drop, args.attn_drop,
                    bn=args.bn, residual=args.residual).to(device)
    if args.model == 'sage':
        model = SAGE(in_feats, args.n_hidden,
                     args.n_hidden, args.n_layers,
                     args.dropout, args.input_drop,
                     bn=args.bn, residual=args.residual).to(device)
    if args.model == 'agdn':
        model = AGDN(in_feats, args.n_hidden,
                     args.n_hidden, args.n_layers,
                     args.n_heads, args.K,
                     args.dropout, args.input_drop, 
                     args.attn_drop, args.edge_drop, args.diffusion_drop,
                     bn=args.bn, output_bn=args.output_bn,
                     transition_matrix=args.transition_matrix,
                     no_dst_attn=args.no_dst_attn,
                     hop_norm=args.hop_norm,
                     weight_style=args.weight_style,
                     pos_emb=not args.no_pos_emb,
                     share_weights=not args.no_share_weights,
                     residual=args.residual,
                     pre_act=args.pre_act).to(device)
    if args.model == 'memagdn':
        model = MemAGDN(in_feats, args.n_hidden,
                     args.n_hidden, args.n_layers,
                     args.n_heads, args.K,
                     args.dropout, args.input_drop, args.attn_drop,
                     in_edge_feats=in_edge_feats).to(device)

    # n_heads = args.n_heads if args.model in ['gat', 'agdn'] else 1
    if args.predictor == 'MLP':
        predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1,
                                args.n_layers, args.dropout).to(device)
    if args.predictor == 'DOT':
        predictor = DotPredictor().to(device)

    if args.predictor == 'COS':
        predictor = CosPredictor().to(device)
    return model, predictor