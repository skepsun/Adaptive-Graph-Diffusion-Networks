cd "$(dirname $0)" 
echo "AGDN+BoT"
python -u ../src/main.py \
    --seed 0 \
    --n-label-iters 1 \
    --lr 0.002 \
    --model agdn \
    --mode teacher \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.75 \
    --input_drop 0.25 \
    --edge_drop 0.35 \
    --attn_drop 0. \
    --diffusion_drop 0. \
    --no-attn-dst \
    --transition-matrix gat_adj \
    --n-epochs 2000 \
    --n-runs 10 \
    --use-labels \
    --mask-rate 0.5 \
    --weight-style HA \
    --checkpoint-path './checkpoint_agdn_bot' 