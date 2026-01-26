#!/bin/bash
#SBATCH --job-name=metrics_plausible_realistic
#SBATCH --output=logs/metrics_plausible_realistic_%j.out
#SBATCH --error=logs/metrics_plausible_realistic_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load cuda/11.8
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

cd /home/kshaltiel/code/CSE-495-Code/prompt_experiments/metrics_scripts
python calculate_metrics_plausible_realistic.py

echo "Metrics calculation for plausible_realistic completed!"
