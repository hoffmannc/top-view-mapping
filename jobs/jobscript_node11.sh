#!/bin/sh
#BSUB -q gpuv100
#BSUB -J top-view-mapping
#BSUB -n 4
#BSUB -gpu "num=4:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s212711@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o top-view-mapping/out/%J.out
#BSUB -e top-view-mapping/out/%J.err
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
#BSUB -m "n-62-20-11"

cd /work3/s212711
module load cuda/12.1.1
source env/bin/activate
cd top-view-mapping
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
    --nproc_per_node=4 \
    --nnodes=5 \
    --node_rank=0 \
    --rdvz_id=456 \
    --rdvz_backend=c10d \
    --rdvz_endpoint=10.66.20.11:4567 \
    train.py config_train