#!/bin/bash
#SBATCH --job-name=clip_all
#SBATCH --output=logs/clip_presence_all_%j.out
#SBATCH --error=logs/clip_presence_all_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load CUDA
module load cuda/11.8

# Activate conda environment
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

# Run CLIP presence scoring for all prompts
cd /home/kshaltiel/code/CSE-495-Code/prompt_experiments
python calculate_clip_presence_all.py

echo "CLIP presence scoring for all prompts completed!"
