#!/usr/bin/env python3
"""
Correlate metrics with nfix (first fixation index) for all prompt experiments
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster

# Metrics to correlate
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

def build_nfix_map(fixfile: Path):
    """Build a mapping from (imageid, category) to average first fixation index (nfix).
    
    The COCO-Search18 dataset contains eye-tracking data showing when people first
    fixated on target objects during visual search. Lower nfix = found quickly (easy),
    higher nfix = found late (hard).
    
    Args:
        fixfile: Path to COCO-Search18 fixations JSON file
        
    Returns:
        dict: mapping (imageid, category) -> average nfix across trials
    """
    print(f"\n{'='*80}")
    print(f"DEBUG: Loading fixation data from {fixfile}")
    print(f"{'='*80}")
    with open(fixfile, "r") as f:
        data = json.load(f)
    
    print(f"DEBUG: Loaded {len(data)} fixation records")

    fmap = {}           # (imageid, category) -> avg nfix
    fmap_trials = {}    # (imageid, category) -> list of nfix values
    processed_count = 0
    found_fixation_count = 0
    
    for rec in data:
        name = rec.get("name")
        if not name:
            continue
        imageid = Path(name).stem
        category = rec.get("task")
        bbox = rec.get("bbox")
        xs = rec.get("X", [])
        ys = rec.get("Y", [])
        
        processed_count += 1
        
        # find first fixation inside bbox
        first_idx = None
        if bbox and xs and ys:
            x0, y0, w, h = bbox
            x1 = x0 + w
            y1 = y0 + h
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                try:
                    if xx >= x0 and xx <= x1 and yy >= y0 and yy <= y1:
                        first_idx = i
                        found_fixation_count += 1
                        break
                except Exception:
                    continue
        
        key = (imageid, category)
        if first_idx is not None:
            fmap_trials.setdefault(key, []).append(first_idx)
        else:
            fmap_trials.setdefault(key, []).append(np.nan)

    print(f"DEBUG: Processed {processed_count} records")
    print(f"DEBUG: Found first fixation in {found_fixation_count} trials")
    
    # compute average per key ignoring NaNs
    for k, lst in fmap_trials.items():
        arr = np.array([x for x in lst if not (isinstance(x, float) and np.isnan(x))], dtype=float)
        if arr.size == 0:
            fmap[k] = np.nan
        else:
            fmap[k] = float(np.nanmean(arr))

    print(f"[OK] Built nfix map with {len(fmap)} image-category keys (averaged across trials)")
    print(f"DEBUG: Sample nfix values (first 5): {dict(list(fmap.items())[:5])}")
    return fmap

def plot_scatter(pairs, metric_name, condition, prompt_name, out_dir, pearson_r):
    """Generate and save a scatter plot of nfix vs metric value."""
    print(f"      DEBUG: Creating scatter plot for {metric_name}, {condition}, n={len(pairs)}")
    
    nfix_vals = [p[0] for p in pairs]
    metric_vals = [p[1] for p in pairs]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(nfix_vals, metric_vals, alpha=0.5, s=20)
    ax.set_xlabel('First Fixation Index (nfix)', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric_name} vs nfix ({condition})\nPearson r = {pearson_r:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plot_dir = Path(out_dir) / "nfix_correlation_plots" / prompt_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f"{prompt_name}_{condition}_{metric_name}.png"
    plot_path = plot_dir / plot_filename
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"      [OK] Saved scatter plot to {plot_path}")


def plot_correlation_bargraph(agg_row, condition, prompt_name, out_dir):
    """Generate and save a bar graph of correlation coefficients for key metrics."""
    print(f"      DEBUG: Creating correlation bar graph for {condition}")
    
    # Select key metrics for bar graph
    key_metrics = ['low_L2', 'mid_L2', 'high_L2', 'rcnn_confidence', 'mid_cosine', 'clip_cosine']
    
    # Filter to only include metrics that have valid correlations
    metrics_to_plot = []
    correlations = []
    for metric in key_metrics:
        if metric in agg_row:
            val = agg_row[metric]
            if not (isinstance(val, float) and np.isnan(val)):
                metrics_to_plot.append(metric)
                correlations.append(val)
    
    if len(metrics_to_plot) == 0:
        print(f"      ⚠️ No valid correlations to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red' if c < 0 else 'green' for c in correlations]
    bars = ax.bar(range(len(metrics_to_plot)), correlations, color=colors, alpha=0.7)
    
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_title(f'Correlation with nfix ({condition}) - {prompt_name}', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)
    
    plot_dir = Path(out_dir) / "nfix_correlation_plots" / prompt_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f"{prompt_name}_{condition}_correlation_bargraph.png"
    plot_path = plot_dir / plot_filename
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"      [OK] Saved correlation bar graph to {plot_path}")


def correlate_prompt_with_nfix(prompt_name, nfix_map, output_dir):
    """
    Correlate metrics with nfix for a single prompt.
    Creates Excel sheets with per-category correlations + overall correlation.
    """
    print(f"\n{'='*80}")
    print(f"DEBUG: Processing prompt={prompt_name}")
    print(f"{'='*80}")
    
    # Paths
    json_dir = Path(f"/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/{prompt_name}")
    
    if not json_dir.exists():
        print(f"  ⚠️ Directory not found: {json_dir}")
        return
    
    # Load all per-image JSONs
    json_files = list(json_dir.glob("*.json"))
    print(f"  Found {len(json_files)} JSON files")
    
    # Build DataFrame with averaged metrics
    data_records = []
    missing_records = []
    
    for json_file in json_files:
        # Parse filename: {category}_{imageid}_{condition}.json
        parts = json_file.stem.split("_")
        if len(parts) < 3:
            continue
        
        category = parts[0]
        imageid = "_".join(parts[1:-1])
        condition = parts[-1]  # bbox or segmentation
        
        # Load metrics and average across reps
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        
        avg_metrics = {}
        for metric_name in METRICS:
            if metric_name in metrics:
                values = metrics[metric_name]
                if len(values) > 0:
                    avg_metrics[metric_name] = float(np.mean(values))
        
        # Look up nfix
        key = (imageid, category)
        nfix = nfix_map.get(key, None)
        
        if nfix is None or (isinstance(nfix, float) and np.isnan(nfix)):
            missing_records.append({"imageid": imageid, "category": category, "condition": condition})
            continue
        
        if len(avg_metrics) > 0:
            record = {
                'category': category,
                'imageid': imageid,
                'condition': condition,
                'nfix': nfix
            }
            record.update(avg_metrics)
            data_records.append(record)
    
    if len(data_records) == 0:
        print(f"  ⚠️ No matching data found")
        return
    
    df = pd.DataFrame(data_records)
    print(f"  Loaded {len(df)} image records with nfix")
    print(f"  Missing {len(missing_records)} records without nfix")
    
    # Get categories
    categories = sorted(df['category'].unique())
    print(f"  Categories: {categories}")
    
    # Create single Excel file with multiple sheets
    output_path = output_dir / f"{prompt_name}_nfix_correlations.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Process each condition (bbox, segmentation) as a separate sheet
        for condition in sorted(df['condition'].unique()):
            print(f"\n  DEBUG: Processing condition={condition}")
            sub = df[df['condition'] == condition]
            print(f"    Found {len(sub)} images for this condition")
            
            # Create correlation table: rows=categories, columns=metrics
            table = pd.DataFrame(index=categories, columns=METRICS, dtype=float)
            
            # Pooled pairs for aggregated correlation
            pooled_pairs = {m: [] for m in METRICS}
            
            # Compute per-category correlations
            for cat in categories:
                subcat = sub[sub['category'] == cat]
                if subcat.empty:
                    continue
                
                print(f"    DEBUG: Category={cat}, {len(subcat)} images")
                
                for metric in METRICS:
                    if metric not in subcat.columns:
                        continue
                    
                    pairs = []
                    for _, r in subcat.iterrows():
                        nfix = r['nfix']
                        val = r.get(metric, np.nan)
                        if not (isinstance(val, float) and np.isnan(val)):
                            pairs.append((nfix, float(val)))
                    
                    if len(pairs) < 2:
                        table.at[cat, metric] = np.nan
                    else:
                        x = np.array([p[0] for p in pairs], dtype=float)
                        y = np.array([p[1] for p in pairs], dtype=float)
                        try:
                            r_val, p_val = pearsonr(x, y)
                            table.at[cat, metric] = float(r_val)
                            pooled_pairs[metric].extend(pairs)
                        except Exception:
                            table.at[cat, metric] = np.nan
            
            # Compute avg row (mean of category correlations)
            avg_row = table.mean(axis=0, skipna=True)
            avg_row.name = "avg"
            
            # Compute aggregated correlation (pooled across all categories)
            agg_row = {}
            for metric, pairs in pooled_pairs.items():
                if len(pairs) < 2:
                    agg_row[metric] = np.nan
                else:
                    x = np.array([p[0] for p in pairs], dtype=float)
                    y = np.array([p[1] for p in pairs], dtype=float)
                    try:
                        r_val, p_val = pearsonr(x, y)
                        agg_row[metric] = float(r_val)
                    except Exception:
                        agg_row[metric] = np.nan
            
            agg_series = pd.Series(agg_row, name="aggregated correlation")
            
            # Generate scatter plots for key metrics: L2 distances, rcnn_confidence, and cosine similarities
            metrics_to_plot = ['low_L2', 'mid_L2', 'high_L2', 'rcnn_confidence', 'mid_cosine', 'clip_cosine']
            for metric_to_plot in metrics_to_plot:
                if metric_to_plot in pooled_pairs and len(pooled_pairs[metric_to_plot]) >= 2:
                    plot_scatter(
                        pooled_pairs[metric_to_plot],
                        metric_to_plot,
                        condition,
                        prompt_name,
                        output_dir,
                        agg_row.get(metric_to_plot, np.nan)
                    )
            
            # Generate bar graph of correlation coefficients
            plot_correlation_bargraph(agg_row, condition, prompt_name, output_dir)
            
            # Append avg and aggregated rows
            avg_df = pd.DataFrame([avg_row.values], index=["avg"], columns=avg_row.index)
            agg_df = pd.DataFrame([agg_series.values], index=["aggregated correlation"], columns=agg_series.index)
            table = pd.concat([table, avg_df, agg_df])
            
            # Write to sheet named by condition
            table.to_excel(writer, sheet_name=condition)
            
            print(f"    ✅ Added sheet: {condition}")
            print(f"       Aggregated correlations: {[(k, f'{v:.3f}') for k, v in list(agg_row.items())[:5]]}")
    
    print(f"  ✅ Saved: {output_path}")

def main():
    print("🎯 Starting nfix correlation for all prompt experiments...")
    
    # Paths
    fixfile = Path("/home/kshaltiel/cluster_test_embed/coco_search18_fixations_TP_train_split1.json")
    output_dir = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_METRICS")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fixation data
    nfix_map = build_nfix_map(fixfile)
    
    # Process each prompt
    prompts = ['minimal', 'contextual', 'realistic', 'natural_setting', 'photorealistic', 'original_quality', 'descriptive',
               'plausible', 'plausible_scene', 'plausible_realistic', 'plausible_setting', 'plausible_placement', 'highly_plausible']
    
    for prompt in prompts:
        correlate_prompt_with_nfix(prompt, nfix_map, output_dir)
    
    print(f"\n{'='*80}")
    print("🎉 All correlations complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    
    print(f"\n{'='*60}")
    print("🎉 All correlations complete!")
    print(f"{'='*60}")
    print(f"Output directory: /home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_METRICS/")
    print(f"Files: {len(prompts)} prompts × 2 conditions = {len(prompts) * 2} correlation files")

if __name__ == "__main__":
    main()
