cd "$(dirname $0)" 
python -u ../../src/main.py \
    --seed 0 \
    --lr 0.002 \
    --model gcn-ha \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 1 \
    --dropout 0.5 \
    --input_drop 0.1 \
    --attn_drop 0.05 \
    --n-epochs 2000 \
    --n-runs 10