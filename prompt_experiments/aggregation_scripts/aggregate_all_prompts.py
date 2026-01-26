#!/usr/bin/env python3
"""
Aggregate per-image JSON metrics into Excel/CSV files for all prompt experiments
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Metrics to include
METRICS = [
    "pixel_L2",
    "low_L2",
    "mid_L2",
    "high_L2",
    "pixel_cosine",
    "low_cosine",
    "mid_cosine",
    "high_cosine",
    "clip_cosine",
    "dino_cosine",
    "clip_L2",
    "dino_L2",
    "rcnn_confidence",
    "yolo_confidence",
]

def compute_stats_from_list(vals):
    """Compute mean, median, min, max, std from a list of values"""
    if not vals:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    arr = np.array(vals, dtype=float)
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "std": np.std(arr)
    }

def aggregate_prompt_metrics(prompt_name):
    """
    Aggregate per-image JSONs for a single prompt into Excel file with bbox/segmentation sheets
    """
    print(f"\n{'='*60}")
    print(f"Processing: {prompt_name}")
    print(f"{'='*60}")
    
    # Paths
    json_dir = Path(f"/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/{prompt_name}")
    output_dir = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_METRICS")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not json_dir.exists():
        print(f"  ⚠️ Directory not found: {json_dir}")
        return
    
    # Find all JSON files
    json_files = list(json_dir.glob("*.json"))
    print(f"  Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print(f"  ⚠️ No JSON files found")
        return
    
    # Group by condition (bbox/segmentation) and category
    condition_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for json_file in json_files:
        # Parse filename: {category}_{imageid}_{condition}.json
        parts = json_file.stem.split("_")
        if len(parts) < 3:
            continue
        
        category = parts[0]
        # imageid could have underscores, condition is last part
        condition = parts[-1]  # bbox or segmentation
        
        # Load metrics
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        
        # Aggregate metrics by category
        for metric_name in METRICS:
            if metric_name in metrics:
                condition_data[condition][category][metric_name].extend(metrics[metric_name])
    
    # Create DataFrames for each condition
    dfs = {}
    for condition in ['bbox', 'segmentation']:
        if condition not in condition_data:
            print(f"  ⚠️ No data for condition: {condition}")
            continue
        
        categories = sorted(condition_data[condition].keys())
        df = pd.DataFrame(index=categories, columns=METRICS, dtype=float)
        
        for category in categories:
            for metric_name in METRICS:
                vals = condition_data[condition][category].get(metric_name, [])
                stats = compute_stats_from_list(vals)
                df.at[category, metric_name] = stats['mean']
        
        dfs[condition] = df
        print(f"  ✓ {condition}: {len(categories)} categories, {len(METRICS)} metrics")
    
    # Save to Excel with multiple sheets
    excel_path = output_dir / f"{prompt_name}_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for condition, df in dfs.items():
            df.to_excel(writer, sheet_name=condition)
    
    print(f"  ✅ Saved: {excel_path}")
    print(f"     Sheets: {list(dfs.keys())}")

def main():
    prompts = ['minimal', 'contextual', 'realistic', 'natural_setting', 'photorealistic', 'original_quality',
               'plausible', 'plausible_scene', 'plausible_realistic', 'plausible_setting', 'plausible_placement', 'highly_plausible']
    
    print("🎯 Starting aggregation for all prompt experiments...")
    print(f"Processing {len(prompts)} prompts")
    
    for prompt in prompts:
        aggregate_prompt_metrics(prompt)
    
    print(f"\n{'='*60}")
    print("🎉 All aggregations complete!")
    print(f"{'='*60}")
    print(f"Output directory: /home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_METRICS/")

if __name__ == "__main__":
    main()
