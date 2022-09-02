cd "$(dirname $0)" 
echo "AGDN"
python -u ../src/main.py \
    --seed 0 \
    --n-label-iters 0 \
    --lr 0.002 \
    --model agdn \
    --mode test \
    --standard-loss \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.75 \
    --input_drop 0.1 \
    --edge_drop 0.15 \
    --attn_drop 0. \
    --diffusion_drop 0. \
    --transition-matrix gat \
    --n-epochs 2000 \
    --n-runs 10 \
    --mask-rate 0. \
    --weight-style HA