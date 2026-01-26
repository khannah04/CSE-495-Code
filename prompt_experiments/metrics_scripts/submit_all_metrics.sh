#!/bin/bash
# Submit all metrics calculation jobs with stagger to avoid HF cache conflicts

echo "Submitting metrics calculation jobs (with 10s stagger)..."

for prompt in minimal contextual realistic natural_setting photorealistic original_quality plausible plausible_scene plausible_realistic plausible_setting plausible_placement highly_plausible; do
    echo "Submitting: $prompt"
    sbatch run_metrics_${prompt}.slurm
    sleep 10
done

echo ""
echo "All jobs submitted! Check status with: squeue -u \$USER"
