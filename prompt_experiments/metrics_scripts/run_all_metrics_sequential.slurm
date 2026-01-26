#!/bin/bash
#SBATCH --job-name=metrics_all_sequential
#SBATCH --output=logs/metrics_all_sequential_%j.out
#SBATCH --error=logs/metrics_all_sequential_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Load CUDA
module load cuda/11.8

# Activate conda environment
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

cd /home/kshaltiel/code/CSE-495-Code/prompt_experiments

# Run all metrics calculations sequentially
for prompt in minimal contextual realistic natural_setting photorealistic original_quality; do
    echo "============================================"
    echo "Starting metrics for: $prompt"
    echo "============================================"
    python calculate_metrics_${prompt}_only.py
    echo ""
    echo "✓ Completed: $prompt"
    echo ""
done

echo "============================================"
echo "ALL METRICS CALCULATIONS COMPLETE!"
echo "============================================"
