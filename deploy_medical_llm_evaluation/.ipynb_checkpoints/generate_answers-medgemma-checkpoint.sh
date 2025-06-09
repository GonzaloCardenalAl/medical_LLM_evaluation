#!/bin/bash

#SBATCH --job-name=final_results
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100_80gb:2
#SBATCH --gres=gpumem:80G
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G

# Load modules
#module load stack/2024-04
#module load stack/2024-06 cuda/12.1.1

unset PYTHONPATH
unset PYTHONHOME
export PATH=~/miniconda3/bin:$PATH
source ~/.bashrc

source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh

export PYTHONNOUSERSITE=1
conda activate vllm_2 || { echo "Failed to activate Conda environment"; exit 1; }

# Debugging
conda list
pip list
module list

# Verify paths before running
echo "Using Python at: $(which python)"
echo "Using torchrun at: $(which torchrun)"
python -c "import openai; print('✅ openai imported successfully')"
python -c "import os; print(dict(os.environ))"
python -c "from transformers import Gemma3ForCausalLM; print('✅importing Gemma3 It works')"

echo "Testing Slurm Variables..."
env | grep SLURM
nvidia-smi

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

scontrol show job $SLURM_JOB_ID

echo "Conda env: $CONDA_PREFIX"
which python
python -c "import transformers; print(transformers.__version__)"

export PYTHONPATH=""
export PYTHONNOUSERSITE=1

#pip uninstall -y transformers torch

# Optional: clean out __pycache__ and any .pyc in case of stale files
#find $CONDA_PREFIX/lib/python3.10/site-packages/transformers -name "*.pyc" -delete
#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/transformers

# Reinstall both freshly
#pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.52.4

# Run model
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    # Single node
    echo "Running in standalone mode..."

    python -c "import transformers; print('Transformers version inside run:', transformers.__version__)"

    echo "Running run_inference.sh..."

    srun --export=ALL ./run_inference-medgemma.sh

#    srun bash -c "
#          source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh && \
#          conda activate vllm_2 && \
#          python -c 'import transformers; print(\"transformers version in srun:\", transformers.__version__)' && \
#          torchrun --standalone --nproc_per_node=${SLURM_NTASKS:-1} get_model_answers_and_prompt_generation.py --model Gemma-3-27B
#        "

else
    # Multi-node
    echo "Running in distributed mode..."

    srun --export=ALL ./run_inference-medgemma.sh

#    srun torchrun \
#        --standalone \
#        --nproc_per_node=${SLURM_NTASKS:-1} \
#        get_model_answers_and_prompt_generation.py --model Gemma-3-27B
    

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