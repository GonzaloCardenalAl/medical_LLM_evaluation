#!/bin/bash

#SBATCH --job-name=f1_score
#SBATCH --output=%x%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=40G

# Load modules
module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
conda activate scispacy_env || { echo "Failed to activate Conda environment"; exit 1; }

unset LD_LIBRARY_PATH

python f1_score.py


conda deactivate
module load stack/2024-06 cuda/12.1.1 python_cuda/3.11.6
source /cluster/home/gcardenal/miniconda3/etc/profile.d/conda.sh
conda activate vllm || { echo "Failed to activate Conda environment"; exit 1; }
python scores.py