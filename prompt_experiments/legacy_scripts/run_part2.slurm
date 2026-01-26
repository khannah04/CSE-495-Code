#!/bin/bash
#SBATCH --job-name=prompt_p2
#SBATCH --output=logs/prompt_part2_%j.out
#SBATCH --error=logs/prompt_part2_%j.err
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

# Run Part 2 (last 3 prompts)
echo ""
echo "========================================"
echo "STARTING PART 2: natural_setting, photorealistic, original_quality"
echo "========================================"
echo ""

python inpainting_prompt_experiments_part2.py

echo ""
echo "========================================"
echo "PART 2 COMPLETE"
echo "========================================"
echo "End time: $(date)"
