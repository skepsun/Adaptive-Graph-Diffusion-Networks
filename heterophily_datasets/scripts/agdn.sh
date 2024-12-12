cd "$(dirname $0)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 
export CUDA_VISIBLE_DEVICES=5 
echo "chameleon"
python -u ../src/main.py \
    --seed 0 \
    --lr 0.002 \
    --wd 2e-4 \
    --model agdn \
    --dataset chameleon \
    --standard-loss \
    --n-layers 3 \
    --n-hidden 1024 \
    --K 3 \
    --n-heads 1 \
    --dropout 0.7 \
    --input_drop 0.7 \
    --edge_drop 0. \
    --attn_drop 0. \
    --diffusion_drop 0. \
    --transition-matrix gat \
    --n-epochs 2000 \
    --patience 2000 \
    --verbose 1 \
    --n-runs 10 \
    --weight-style HA
echo "squirrel"
python -u ../src/main.py \
    --seed 0 \
    --lr 0.005 \
    --wd 1e-4 \
    --model agdn \
    --dataset squirrel \
    --standard-loss \
    --n-layers 3 \
    --n-hidden 1024 \
    --K 3 \
    --n-heads 1 \
    --dropout 0.7 \
    --input_drop 0.7 \
    --edge_drop 0. \
    --attn_drop 0. \
    --diffusion_drop 0. \
    --transition-matrix gat \
    --n-epochs 2000 \
    --patience 2000 \
    --verbose 1 \
    --n-runs 10 \
    --weight-style HA
echo "actor"
python -u ../src/main.py \
    --seed 0 \
    --lr 0.005 \
    --wd 5e-3 \
    --model agdn \
    --dataset actor \
    --standard-loss \
    --n-layers 3 \
    --n-hidden 1024 \
    --K 3 \
    --n-heads 1 \
    --dropout 0.4 \
    --input_drop 0.4 \
    --edge_drop 0. \
    --attn_drop 0. \
    --diffusion_drop 0. \
    --transition-matrix gat \
    --n-epochs 2000 \
    --patience 2000 \
    --verbose 1 \
    --n-runs 10 \
    --weight-style HA