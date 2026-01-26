#!/bin/bash
#SBATCH --job-name=plausible_scene
#SBATCH --output=logs/plausible_scene_%j.out
#SBATCH --error=logs/plausible_scene_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module load cuda/11.8
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

echo ""
echo "=========================================="
echo "STARTING: plausible_scene"
echo "=========================================="
echo ""

python inpainting_plausible_scene.py

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End time: $(date)"
