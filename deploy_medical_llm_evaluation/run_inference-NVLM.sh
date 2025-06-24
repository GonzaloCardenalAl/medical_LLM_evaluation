#!/bin/bash
# File: run_inference-NVLM.sh

unset PYTHONPATH
unset PYTHONHOME

# === PATH SETUP ===
export PATH="$HOME/miniconda3/bin:$PATH"

pip uninstall -y transformers

# Ensure all remnants are gone
rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/transformers
rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/transformers-*.dist-info

# === LOAD CONDA ===
# ~/.bashrc may not work in non-interactive shells, so be explicit:
source "$HOME/miniconda3/etc/profile.d/conda.sh"

source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
conda activate transformers_llm || { echo "Failed to activate conda env"; exit 1; }

export PATH="$CONDA_PREFIX/bin:$PATH"

# Inside your conda env (vllm_2), uninstall old torch
pip uninstall -y torch

# Then install PyTorch 2.5 (with CUDA 12.1, assuming your A100 GPUs)
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers

echo "Inside srun:"
which python
python -c "import sys; print('✅ Python version:', sys.version)"
python -c "import transformers; print('✅ Transformers version:', transformers.__version__)"
python -c "import torch; print(torch.__version__)"

torchrun --standalone --nproc_per_node=${SLURM_NTASKS:-1} get_model_answers_and_prompt_generation.py --model NVLM 