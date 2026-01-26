#!/bin/bash
#SBATCH --job-name=inpaint_test
#SBATCH --output=logs/inpaint_test_%j.out
#SBATCH --error=logs/inpaint_test_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module load cuda/11.8 # adjust to cluster modules
#source /home/kshaltiel/miniconda3/envs/inpaint_env_py39/bin/activate
source /home/kshaltiel/miniconda3/etc/profile.d/conda.sh
conda activate inpaint_env_py39

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the full pipeline
echo ""
echo "========================================"
echo "STARTING PROMPT EXPERIMENTS PIPELINE"
echo "========================================"
echo ""

# Step 1: Generate images with different prompts
echo "Step 1: Generating images with different prompt strategies..."
python inpainting_prompt_experiments.py

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo "End time: $(date)"
