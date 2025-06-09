#!/bin/bash
# File: run_scores.sh


source "$HOME/miniconda3/etc/profile.d/conda.sh"

source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
conda activate vllm_2 || { echo "Failed to activate conda env"; exit 1; }

export PATH="$CONDA_PREFIX/bin:$PATH"

python scores.py