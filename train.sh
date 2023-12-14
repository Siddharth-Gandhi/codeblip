#!/bin/bash
#SBATCH --job-name=s2_train
#SBATCH --output=logs/stage1_train/s2_train_%A.log
#SBATCH --partition=gpu
#SBATCH --exclude=boston-2-25,boston-2-27,boston-2-29,boston-2-31
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codeblip

echo "On host $(hostname)"
nvidia-smi

# Run the training script
# python src/train.py
python src/stage2_train.py