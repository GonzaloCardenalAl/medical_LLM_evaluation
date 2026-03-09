#!/bin/bash
#SBATCH --job-name=llama8b_rag
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=quadro_rtx_6000:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G


set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
VENV="/cluster/scratch/gcardenal/venvs/transformers_llm_rag"
SCRIPT="/cluster/home/gcardenal/HIV/medical_llm_evaluation/deploy_medical_llm_evaluation/get_model_answers_and_prompt_generation_plus_rag.py"

# -----------------------------
# Clean environment
# -----------------------------
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# Activate uv venv
source "${VENV}/bin/activate"

echo "Using python: $(which python)"
python -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable)"

# -----------------------------
# Network proxy (Euler/ETH style)
# -----------------------------
export http_proxy="http://proxy.service.consul:3128"
export https_proxy="http://proxy.service.consul:3128"
export no_proxy="api.wandb.ai,app.neptune.ai,jupyter.eu-dev.hpc.ethz.ch,jupyter-staging.euler.hpc.ethz.ch,jupyter.euler.hpc.ethz.ch,.consul,localhost,127.0.0.1,127.0.0.0/8,169.254.0.0/16"

# -----------------------------
# API keys (recommended: export these before sbatch)
# -----------------------------
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
: "${PINECONE_API_KEY:?PINECONE_API_KEY is not set}"

echo "OPENAI_API_KEY present: yes"
echo "PINECONE_API_KEY present: yes"

# -----------------------------
# Debug hardware / libs
# -----------------------------
nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'n_gpus:', torch.cuda.device_count())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import pinecone; print('pinecone:', pinecone.__version__)"
python -c "from openai import OpenAI; print('openai client ok')"

# -----------------------------
# Run
# -----------------------------
echo "Running model pipeline..."
#srun --export=ALL python "${SCRIPT}" --model MedGemma-3-27B-RAG
srun --export=ALL python "${SCRIPT}" --model Llama-8B-RAG