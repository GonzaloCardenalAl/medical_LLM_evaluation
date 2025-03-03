#!/bin/bash

#SBATCH --job-name=llama_no_api
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100_80gb:3
#SBATCH --gres=gpumem:80G
#SBATCH --time=8:00:00
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
        get_model_answers_and_prompt_generation.py --model NVLM

else
    # Multi-node
    echo "Running in distributed mode..."

    srun torchrun \
        --standalone \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        get_model_answers_and_prompt_generation.py --model NVLM
        
    
fi

#conda deactivate
#module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
#source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
#conda activate scispacy_env || { echo "Failed to activate Conda environment"; exit 1; }
#unset LD_LIBRARY_PATH
#python f1_score.py


#conda deactivate
#module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
#source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
#conda activate vllm || { echo "Failed to activate Conda environment"; exit 1; }
#python scores.py