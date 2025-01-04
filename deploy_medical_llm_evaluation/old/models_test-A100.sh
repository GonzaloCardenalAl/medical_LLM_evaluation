#!/bin/bash

#SBATCH --job-name=llama70b_test
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100_80gb:2
#SBATCH --gres=gpumem:80G
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10G


# Load modules
#module load stack/2024-04
module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
conda activate vllm || { echo "Failed to activate Conda environment"; exit 1; }

# Debugging
conda list
pip list
module list


echo "Testing Slurm Variables..."
env | grep SLURM
nvidia-smi

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

scontrol show job $SLURM_JOB_ID

# Run model
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    # Single node
    echo "Running in standalone mode..."
    srun torchrun \
        --standalone \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        trial.py
else
    # Multi-node
    echo "Running in distributed mode..."
    srun torchrun \
        --nnodes=${SLURM_JOB_NUM_NODES:-1} \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        --rdzv_id=$RANDOM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$head_node_ip:29500 \
        trial.py
fi