#!/bin/bash

#SBATCH --job-name=GPT_score
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=40G

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

echo "Conda env: $CONDA_PREFIX"
which python

# Trace commands
set -x

# Run script
echo ">>> Running scores-unsupervised.py"
python scores-unsupervised.py || { echo "❌ scores-unsupervised.py crashed"; exit 1; }
echo "✅ scores-unsupervised.py finished"