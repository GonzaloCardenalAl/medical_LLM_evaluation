#!/bin/bash

#SBATCH --job-name=nvllm_test
#SBATCH --output=%x%j.out
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=2
#SBATCH --time=00:20:00

module load mamba
source activate nvllm

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Analytical code
srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $SLURM_GPUS_PER_NODE \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29521 \
main_nvllm.py