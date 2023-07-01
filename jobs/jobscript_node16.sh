#!/bin/sh
#BSUB -q gpuv100
#BSUB -J top-view-mapping
#BSUB -n 2
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s212711@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o top-view-mapping/out/%J.out
#BSUB -e top-view-mapping/out/%J.err
#BSUB -m "n-62-20-16"

cd /work3/s212711
module load cuda/12.1.1
source env/bin/activate
cd top-view-mapping
export CUDA_VISIBLE_DEVICES=0,1
torchrun \
    --nproc_per_node=2 \
    --nnodes=4 \
    --node_rank=3 \
    --rdzv-id=456 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=10.66.20.13:4567 \
    train.py config_train