#!/bin/bash
#SBATCH --job-name=clip_presence
#SBATCH --output=logs/clip_presence_%j.out
#SBATCH --error=logs/clip_presence_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load CUDA
module load cuda/11.8

# Activate conda environment
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

# Run CLIP presence scoring
cd /home/kshaltiel/code/CSE-495-Code/prompt_experiments
python calculate_clip_presence.py

echo "CLIP presence scoring completed!"
