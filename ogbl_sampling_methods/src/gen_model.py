from models import GCN, GAT, SAGE, AGDN, DotPredictor, LinkPredictor

def gen_model(args, in_feats, in_edge_feats, device):
    if args.model == 'gcn':
        model = GCN(in_feats, args.n_hidden,
                    args.n_hidden, args.n_layers,
                    args.dropout, args.input_drop, in_edge_feats=in_edge_feats,
                    bn=args.bn, residual=args.residual).to(device)
    if args.model == 'gat':
        model = GAT(in_feats, args.n_hidden,
                    args.n_hidden, args.n_layers,
                    args.n_heads,
                    args.dropout, args.input_drop, args.attn_drop,
                    in_edge_feats=in_edge_feats,
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
                     args.dropout, args.input_drop, args.attn_drop,
                     pos_emb=not args.no_pos_emb,
                     in_edge_feats=in_edge_feats, 
                     residual=args.residual).to(device)

    n_heads = args.n_heads if args.model in ['gat', 'agdn'] else 1
    # predictor = DotPredictor(args.n_hidden * n_heads, args.n_hidden * n_heads, 1,
    #                           args.n_layers, args.dropout).to(device)
    predictor = LinkPredictor(args.n_hidden * n_heads, args.n_hidden * n_heads, 1,
                              args.n_layers, args.dropout).to(device)
    return model, predictor