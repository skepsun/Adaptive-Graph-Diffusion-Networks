cd "$(dirname $0)" 
python -u ../../src/main.py \
    --seed 0 \
    --lr 0.002 \
    --model gat-ha \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.5 \
    --input_drop 0.25 \
    --edge_drop 0. \
    --attn_drop 0.05 \
    --norm sym \
    --no-attn-dst \
    --n-epochs 2000 \
    --n-runs 10 \
    --gpu 0