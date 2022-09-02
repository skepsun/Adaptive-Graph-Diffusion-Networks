cd "$(dirname $0)"
python -u ../src/main.py \
    --dataset ogbl-citation2 \
    --model agdn \
    --no-pos-emb \
    --K 2 \
    --weight-style HA \
    --batch-size 128 \
    --sampler saint_rw \
    --n-subgraphs 1000 \
    --rw-budget 25000 3 \
    --neighbor-size 15 10 5 \
    --lr 0.001 \
    --negative-sampler persource \
    --n-neg 1 \
    --eval-steps 10 \
    --eval-from 1000 \
    --epochs 2000 \
    --bn 