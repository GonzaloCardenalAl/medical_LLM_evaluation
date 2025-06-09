#!/bin/bash

#SBATCH --job-name=2final_results_deepseekr1&llama4
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100-pcie-40gb:4
#SBATCH --gres=gpumem:40G
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G

# Load modules
#module load stack/2024-04
#module load stack/2024-06 cuda/12.1.1
source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh

export PYTHONNOUSERSITE=1
conda activate vllm || { echo "Failed to activate Conda environment"; exit 1; }

# Debugging
conda list
pip list
module list

# Verify paths before running
echo "Using Python at: $(which python)"
echo "Using torchrun at: $(which torchrun)"
python -c "import openai; print('✅ openai imported successfully')"

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
    RANK_LOG=rank$SLURM_PROCID.log
    $(which torchrun) --nproc_per_node=4 get_model_answers_and_prompt_generation.py --model Gemma-3-27B
    

else
    # Multi-node
    echo "Running in distributed mode..."

    srun torchrun \
        --standalone \
        --nproc_per_node=${SLURM_NTASKS:-1} \
        get_model_answers_and_prompt_generation.py --model Gemma-3-27B
    

fi

#get_model_answers_and_prompt_generation.py --model Llama Meditron Claude Med42 Llama-1B Llama-8B NVLM
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