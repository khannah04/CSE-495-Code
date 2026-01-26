#!/bin/bash
# Wait for plausible_scene inpainting job (8364) to finish, then submit metrics

echo "Waiting for plausible_scene job (8364) to complete..."
echo "Started at: $(date)"

# Check every 5 minutes
while true; do
    # Check if job 8364 is still running
    if squeue -j 8364 2>/dev/null | grep -q "8364"; then
        echo "[$(date +%H:%M:%S)] Job 8364 still running... checking again in 5 minutes"
        sleep 300  # 5 minutes
    else
        echo "[$(date +%H:%M:%S)] Job 8364 completed!"
        break
    fi
done

echo "Waiting 30 seconds to ensure job cleanup..."
sleep 30

echo "Submitting metrics for plausible_scene..."
cd /home/kshaltiel/code/CSE-495-Code/prompt_experiments/metrics_scripts
sbatch run_metrics_plausible_scene.slurm

echo "Done! Submitted at: $(date)"
