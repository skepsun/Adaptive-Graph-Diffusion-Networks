cd "$(dirname $0)" 
echo "AGDN+XRT+BoT"
python -u ../src/main.py \
    --seed 0 \
    --n-label-iters 1 \
    --lr 0.01 \
    --use-xrt-emb \
    --model agdn \
    --mode teacher \
    --n-layers 2 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.85 \
    --input_drop 0.35 \
    --edge_drop 0.6 \
    --attn_drop 0. \
    --diffusion_drop 0.3 \
    --transition-matrix gat_sym \
    --n-epochs 2000 \
    --n-runs 10 \
    --use-labels \
    --weight-style HA \
    --checkpoint-path './checkpoint_agdn_bot_xrt' 
