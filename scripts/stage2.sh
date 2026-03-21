NUM_GPUS=1
NUM_CPUS_PER_GPU=8
BATCH_SIZE_PER_GPU=64

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23950 # random port

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 10 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    --verbose 2 \
    --input_size 224 \
    --lr 1e-4 \
    --hf_split_train "train" \
    --huggingface_train_repo "OwensLab/CommunityForensics-Small" \
    --exp_name "example" \
    --prototype_path "./prototype/pca_3type_num64.pt" \
    --fe_path "./pretrained/stage1.pt" \
