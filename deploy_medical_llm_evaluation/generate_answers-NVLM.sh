#!/bin/bash

#SBATCH --job-name=final_results
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:2
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
conda activate transformers_llm || { echo "Failed to activate Conda environment"; exit 1; }

# Debugging
conda list
pip list
module list

# Verify paths before running
echo "Using Python at: $(which python)"
echo "Using torchrun at: $(which torchrun)"
python -c "import openai; print('✅ openai imported successfully')"
python -c "import os; print(dict(os.environ))"

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

pip install transformers==4.52.4

# Run model
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    # Single node
    echo "Running in standalone mode..."

    python -c "import transformers; print('Transformers version inside run:', transformers.__version__)"

    echo "Running run_inference.sh..."

    srun --export=ALL ./run_inference-NVLM.sh


else
    # Multi-node
    echo "Running in distributed mode..."

    srun --export=ALL ./run_inference-NVLM.sh


fi