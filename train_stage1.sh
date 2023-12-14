#!/bin/bash -x
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128G
###SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=codeblip_llama
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyushkh@andrew.cmu.edu
#SBATCH --partition=babel-shared-long

# Activate the conda environment
source /data/tir/projects/tir6/general/piyushkh/conda/bin/activate codeblip

echo "On host $(hostname)"
nvidia-smi

# Run the training script
python src/train.py > stage1_512_output.txt