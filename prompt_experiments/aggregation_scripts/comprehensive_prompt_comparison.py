"""
Comprehensive analysis comparing all 12 prompt strategies.

Analyzes:
1. Object detection success (R-CNN, YOLO confidence)
2. Object presence preservation (CLIP presence scores)
3. Visual quality (LaRE, DINO, CLIP similarity)
4. Correlation with nfix (human behavior prediction)
5. Inter-prompt similarity (how similar are different prompts)

Outputs:
- Summary CSV with aggregate metrics per prompt
- Per-prompt Excel files with correlation tables (categories × metrics)
- Visualization plots
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
METRICS_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS")
NFIX_JSON = Path("/home/kshaltiel/cluster_test_embed/coco_search18_fixations_TP_train_split1.json")
OUTPUT_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_ANALYSIS")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# All prompts to analyze
PROMPTS = [
    "minimal", "contextual", "realistic", "natural_setting", 
    "photorealistic", "original_quality", "descriptive", "plausible", "plausible_scene",
    "plausible_realistic", "plausible_setting", "plausible_placement", 
    "highly_plausible"
]

CATEGORIES = [
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard",
    "knife", "laptop", "microwave", "mouse", "oven", "potted plant",
    "sink", "stop sign", "toilet", "tv"
]

# Metric groups for organized analysis
DETECTION_METRICS = ["rcnn_confidence", "yolo_confidence"]
PRESENCE_METRICS = ["clip_presence_prob", "clip_presence_logit_diff", "clip_text_similarity"]
QUALITY_METRICS = ["lare", "dino_cosine", "clip_cosine", "jenga_score"]
PIXEL_METRICS = ["pixel_l2", "pixel_cosine"]
ALEXNET_METRICS = ["alex_low_l2", "alex_low_cosine", "alex_mid_l2", "alex_mid_cosine", 
                   "alex_high_l2", "alex_high_cosine"]


# ==============================================================================
# PARSE FILENAME TO EXTRACT METADATA
# ==============================================================================
# Metric JSON files follow naming convention: category_taskid_masktype.json
# - category: object type (e.g., "bottle", "potted plant")
# - task_id: COCO image ID (12-digit number like 000000123456)
# - mask_type: "bbox" (bounding box) or "segmentation" (precise mask)
#
# This parsing is critical for matching images to nfix data and grouping results
# ==============================================================================
def parse_filename(fname: str):
    """Parse per-image metric filenames to extract metadata.
    
    Input format: 'category_taskid_masktype.json'
    Example: 'bottle_000000123456_bbox.json' or 'potted plant_000000123456_segmentation.json'
    
    Why we parse filenames:
    - task_id links to COCO-Search18 eye-tracking data for nfix values
    - category enables per-category correlation analysis
    - mask_type separates bbox vs segmentation analysis
    
    Returns:
        tuple: (category, task_id, mask_type)
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    
    # Find the task_id (long numeric string)
    task_id_idx = None
    for i, part in enumerate(parts):
        if len(part) >= 8 and part.isdigit():
            task_id_idx = i
            break
    
    if task_id_idx is None or task_id_idx == 0:
        # Fallback
        return parts[0], parts[1] if len(parts) > 1 else "", "_".join(parts[2:])
    
    category = "_".join(parts[:task_id_idx])
    task_id = parts[task_id_idx]
    mask_type = "_".join(parts[task_id_idx + 1:])
    
    return category, task_id, mask_type


# ==============================================================================
# LOAD NFIX DATA (HUMAN VISUAL SEARCH DIFFICULTY)
# ==============================================================================
# nfix = number of fixations before finding target object
# Lower nfix = easier to find (visually salient, distinct)
# Higher nfix = harder to find (blends in, complex scene)
#
# This is our GROUND TRUTH for how humans perceive these images
# Strong metric-nfix correlations mean our metrics predict human behavior
#
# Data comes from COCO-Search18 eye-tracking dataset where participants
# searched for specific objects in COCO images while their eye movements
# were recorded. We use this to evaluate if inpainted images preserve
# the same visual search difficulty as originals.
# ==============================================================================
def build_nfix_map(fixfile: Path):
    """Build mapping from (task_id, category) to average nfix.
    
    Process:
    1. Load COCO-Search18 fixation data (eye-tracking recordings)
    2. For each trial, find first fixation that lands inside target bbox
    3. Average across multiple trials for same image to get stable nfix
    
    Why average across trials?
    - Same image shown to multiple participants
    - Individual variation in search strategy
    - Averaging gives more robust difficulty estimate
    
    Returns:
        dict: (task_id, category) -> average nfix value
    """
    print(f"\nLoading nfix data from {fixfile}...")
    with open(fixfile) as f:
        data = json.load(f)
    
    nfix_trials = {}  # (task_id, category) -> list of nfix values
    
    for rec in data:
        name = rec.get("name")
        if not name:
            continue
        
        task_id = Path(name).stem
        category = rec.get("task")
        bbox = rec.get("bbox")
        xs = rec.get("X", [])
        ys = rec.get("Y", [])
        
        # Find first fixation inside bbox
        first_idx = None
        if bbox and xs and ys:
            x0, y0, w, h = bbox
            x1, y1 = x0 + w, y0 + h
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                if x0 <= xx <= x1 and y0 <= yy <= y1:
                    first_idx = i
                    break
        
        key = (task_id, category)
        if first_idx is not None:
            nfix_trials.setdefault(key, []).append(first_idx)
    
    # Average across trials
    nfix_map = {}
    for key, vals in nfix_trials.items():
        arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        if arr.size > 0:
            nfix_map[key] = float(np.mean(arr))
    
    print(f"  Loaded nfix for {len(nfix_map)} unique (task_id, category) pairs")
    return nfix_map


# ==============================================================================
# LOAD PER-IMAGE METRICS FOR A PROMPT
# ==============================================================================
# Each JSON file contains 10 repetitions (rep0-rep9) of inpainting
# We average across reps to get stable metric estimates
#
# Why 10 reps? Inpainting has randomness (diffusion noise), so we run multiple
# times and average to get reliable measurements that aren't affected by lucky/unlucky noise
# ==============================================================================
def load_per_image_dir(prompt_dir: Path, nfix_map: dict):
    """Load all per-image JSON files and average metrics across reps.
    
    JSON structure:
        {
            "pixel_l2": [val0, val1, ..., val9],      # 10 repetitions
            "clip_cosine": [val0, val1, ..., val9],
            "rcnn_confidence": [val0, ..., val9],
            ...
        }
    
    Processing:
    1. Parse filename to get category, task_id, mask_type
    2. Load metric arrays from JSON
    3. Average each metric across 10 reps → metric_mean
    4. Look up nfix value from eye-tracking data
    5. Compute jenga_score = clip_cosine × dino_cosine
    
    Why jenga_score?
    - Combines perceptual (CLIP) and semantic (DINO) preservation
    - Multiplication ensures both must be high (not just one)
    - Higher score = better overall preservation
    
    Returns:
        DataFrame with columns: category, task_id, mask_type, nfix, <metric>_mean
    """
    print(f"\nLoading metrics from: {prompt_dir}")
    rows = []
    
    for json_file in sorted(prompt_dir.glob("*.json")):
        category, task_id, mask_type = parse_filename(json_file.name)
        
        if mask_type not in ["bbox", "segmentation"]:
            continue
        
        # Load metrics
        with open(json_file) as f:
            metrics = json.load(f)
        
        # Get nfix
        nfix_key = (task_id, category)
        nfix = nfix_map.get(nfix_key)
        
        # Build row
        row = {
            "category": category,
            "task_id": task_id,
            "mask_type": mask_type,
            "nfix": nfix
        }
        
        # Average all metrics across reps
        for metric_key, metric_values in metrics.items():
            if isinstance(metric_values, list) and len(metric_values) > 0:
                row[f"{metric_key}_mean"] = np.mean(metric_values)
        
        # Compute jenga_score
        if "clip_cosine_mean" in row and "dino_cosine_mean" in row:
            row["jenga_score_mean"] = row["clip_cosine_mean"] * row["dino_cosine_mean"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} images")
    if len(df) > 0:
        print(f"  - bbox: {(df['mask_type'] == 'bbox').sum()}")
        print(f"  - segmentation: {(df['mask_type'] == 'segmentation').sum()}")
        print(f"  - with nfix: {df['nfix'].notna().sum()}")
    
    return df


# ==============================================================================
# COMPUTE CORRELATIONS: CATEGORY × METRIC TABLE
# ==============================================================================
# For each mask_type, create a table with:
# - Rows: categories (bottle, chair, etc.)
# - Columns: metrics (pixel_l2, clip_cosine, lare, etc.)
# - Cells: Pearson r correlation between metric and nfix for that category
# - Extra rows: 
#   * "avg" = mean of category r's (average correlation across categories)
#   * "aggregated correlation" = Pearson r pooling ALL images together
#
# Why two summary rows?
# - "avg": Each category weighted equally (good for seeing general trends)
# - "aggregated": Weighted by sample size (more images = more influence)
#
# Interpretation:
# - Positive r: metric increases → nfix increases (harder to find)
# - Negative r: metric increases → nfix decreases (easier to find)
# - |r| > 0.3: meaningful correlation
# - |r| > 0.5: strong correlation
# ==============================================================================
def compute_correlation_table(df, mask_type, categories):
    """Compute correlation table for one mask type.
    
    Process:
    1. Filter to specific mask_type (bbox or segmentation) with nfix data
    2. For each category:
       - Collect (nfix, metric_value) pairs for all images
       - Compute Pearson correlation
       - Store in table cell
    3. Compute "avg" row: mean of category correlations
    4. Compute "aggregated" row: correlation pooling all categories together
    
    Why per-category analysis?
    - Some metrics may work better for certain object types
    - Identifies if prompt excels at specific categories
    - Reveals metric biases (e.g., works for large objects only)
    
    Returns:
        DataFrame with rows=categories+avg+aggregated, columns=metrics
        Returns None if insufficient data (<10 images with nfix)
    """
    # Filter to this mask type and images with nfix
    sub = df[(df["mask_type"] == mask_type) & (df["nfix"].notna())].copy()
    
    if len(sub) < 10:
        return None
    
    # Get metric columns
    metric_cols = sorted([c for c in sub.columns if c.endswith("_mean") and c not in ["nfix"]])
    metric_names = [c.replace("_mean", "") for c in metric_cols]
    
    # Create table
    table = pd.DataFrame(index=categories, columns=metric_names, dtype=float)
    
    # Pooled pairs for aggregated correlation
    pooled_pairs = {m: [] for m in metric_names}
    
    # Compute per-category correlations
    for cat in categories:
        subcat = sub[sub["category"] == cat]
        
        if len(subcat) < 2:
            continue
        
        for mcol, mname in zip(metric_cols, metric_names):
            pairs = []
            
            for _, row in subcat.iterrows():
                nfix_val = row["nfix"]
                metric_val = row.get(mcol)
                
                if pd.notna(nfix_val) and pd.notna(metric_val):
                    pairs.append((float(nfix_val), float(metric_val)))
            
            if len(pairs) < 2:
                table.at[cat, mname] = np.nan
            else:
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                # Check if either array is constant (no variation)
                if np.std(x) == 0 or np.std(y) == 0:
                    table.at[cat, mname] = np.nan
                else:
                    try:
                        r_val, _ = stats.pearsonr(x, y)
                        table.at[cat, mname] = float(r_val)
                        pooled_pairs[mname].extend(pairs)
                    except:
                        table.at[cat, mname] = np.nan
    
    # Compute avg row (mean of category r's)
    avg_row = table.mean(axis=0, skipna=True)
    avg_row.name = "avg"
    
    # Compute aggregated correlation (pooled across categories)
    agg_row = {}
    for mname, pairs in pooled_pairs.items():
        if len(pairs) < 2:
            agg_row[mname] = np.nan
        else:
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            # Check if either array is constant (no variation)
            if np.std(x) == 0 or np.std(y) == 0:
                agg_row[mname] = np.nan
            else:
                try:
                    r_val, _ = stats.pearsonr(x, y)
                    agg_row[mname] = float(r_val)
                except:
                    agg_row[mname] = np.nan
    
    agg_series = pd.Series(agg_row, name="aggregated correlation")
    
    # Append avg and aggregated rows
    avg_df = pd.DataFrame([avg_row.values], index=["avg"], columns=avg_row.index)
    agg_df = pd.DataFrame([agg_series.values], index=["aggregated correlation"], columns=agg_series.index)
    table = pd.concat([table, avg_df, agg_df])
    
    return table


# ==============================================================================
# COMPUTE SUMMARY STATISTICS
# ==============================================================================
# Aggregate all images for a prompt+mask_type into single summary metrics
# This gives us a "report card" for each prompt strategy
#
# Key Questions Answered:
# 1. Detection Success: Are objects recognizable by R-CNN/YOLO?
# 2. Presence Preservation: Does CLIP still detect the target object?
# 3. Quality: How natural/realistic are results? (LaRE, DINO, CLIP)
# 4. Consistency: How variable are the results? (std dev)
#
# Use Case: Quickly compare prompts without drilling into per-category details
# ==============================================================================
def compute_summary_statistics(df, prompt_name, mask_type):
    """Compute aggregate statistics for a prompt+mask_type combination.
    
    Metrics Computed:
    
    Detection Success:
    - rcnn/yolo_detection_rate: % images with confidence > 0.5
      (What fraction of images have detectable objects?)
    - rcnn/yolo_mean_confidence: average detection confidence
      (How confident are the detectors on average?)
    
    Presence Preservation:
    - clip_presence_prob_mean: avg CLIP probability target is present
      (Does CLIP think the target object is still there?)
    - clip_presence_logit_diff_mean: avg confidence in presence vs absence
      (How strongly does CLIP detect the object?)
    
    Quality Metrics:
    - lare_mean/median: Latent Reconstruction Error
      Lower = more natural (diffusion model easily reconstructs)
    - dino_cosine_mean: semantic similarity to original
      Higher = better preservation of object shape/structure
    - clip_cosine_mean: perceptual similarity to original
      Higher = looks more similar to humans
    - jenga_score_mean: combined CLIP × DINO preservation
      Higher = robust preservation across both dimensions
    
    Returns:
        dict with summary statistics, or None if no data
    """
    sub = df[df["mask_type"] == mask_type].copy()
    
    if len(sub) == 0:
        return None
    
    summary = {
        "prompt": prompt_name,
        "mask_type": mask_type,
        "total_images": len(sub),
        "images_with_nfix": sub["nfix"].notna().sum()
    }
    
    # Detection success
    for metric in ["rcnn_confidence", "yolo_confidence"]:
        if f"{metric}_mean" in sub.columns:
            summary[f"{metric}_detection_rate"] = (sub[f"{metric}_mean"] > 0.5).mean()
            summary[f"{metric}_mean_confidence"] = sub[f"{metric}_mean"].mean()
    
    # Presence preservation
    for metric in ["clip_presence_prob", "clip_presence_logit_diff"]:
        if f"{metric}_mean" in sub.columns:
            summary[f"{metric}_mean"] = sub[f"{metric}_mean"].mean()
            summary[f"{metric}_std"] = sub[f"{metric}_mean"].std()
    
    # Quality metrics
    for metric in ["lare", "dino_cosine", "clip_cosine", "jenga_score"]:
        if f"{metric}_mean" in sub.columns:
            summary[f"{metric}_mean"] = sub[f"{metric}_mean"].mean()
            summary[f"{metric}_median"] = sub[f"{metric}_mean"].median()
            summary[f"{metric}_std"] = sub[f"{metric}_mean"].std()
    
    # Pixel metrics
    for metric in ["pixel_l2", "pixel_cosine"]:
        if f"{metric}_mean" in sub.columns:
            summary[f"{metric}_mean"] = sub[f"{metric}_mean"].mean()
    
    return summary


# ==============================================================================
# COMPUTE INTER-PROMPT CORRELATION MATRIX
# ==============================================================================
# This answers: "Are different prompts actually producing different results?"
#
# Method:
# 1. Each prompt has a "signature": vector of aggregated correlations for all metrics
#    Example: minimal = [pixel_l2: +0.4, dino: -0.2, lare: +0.3, ...]
# 2. Correlate these signature vectors between prompts
# 3. High correlation = prompts behave similarly
#
# Use Cases:
# - Identify redundant prompts (r > 0.9): picking one is sufficient
# - Find complementary prompts (r < 0.5): both provide unique insights
# - Cluster prompts into families (e.g., all "plausible" variants group together)
#
# Interpretation:
# - r = 1.0: Identical metric-nfix patterns (diagonal)
# - r > 0.8: Very similar prompts, likely redundant
# - r = 0.0: Uncorrelated, prompts work differently
# - r < 0: Opposing patterns (rare, but possible)
# ==============================================================================
def compute_inter_prompt_correlation(all_correlations):
    """Compare prompts based on their correlation patterns.
    
    What this measures:
    - NOT: "Do prompts produce visually similar images?"
    - YES: "Do prompts have similar metric-nfix relationship patterns?"
    
    Example:
    If "minimal" and "contextual" have inter-prompt r=0.95:
    - Both produce images where same metrics predict nfix similarly
    - Likely redundant - picking one is sufficient
    
    If "plausible" and "realistic" have r=0.40:
    - Different approaches to inpainting
    - Both worth keeping for diversity
    
    Returns:
        Correlation matrix (DataFrame): prompts × prompts with correlation values
        Returns None if insufficient data
    """
    # Build DataFrame: rows=metrics, columns=prompts, values=aggregated_r
    # OPTION 1: Use only aggregated row (current - fast but less complete)
    # data = {}
    # for prompt, corr_data in all_correlations.items():
    #     if corr_data is None:
    #         continue
    #     agg_row = corr_data.iloc[-1]
    #     data[prompt] = agg_row
    
    # OPTION 2: Use ALL values (all categories + avg + aggregated)
    data = {}
    for prompt, corr_data in all_correlations.items():
        if corr_data is None:
            continue
        # Flatten entire correlation table into a single vector
        # This includes all 18 categories + avg + aggregated for all metrics
        all_values = corr_data.values.flatten()
        # Create index like "category_metric" for each value
        indices = [f"{row}_{col}" for row in corr_data.index for col in corr_data.columns]
        data[prompt] = pd.Series(all_values, index=indices)
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    # Remove rows where any prompt has NaN (can't correlate)
    df = df.dropna()
    
    # Compute correlation between prompt columns
    corr_matrix = df.corr(method="pearson")
    
    return corr_matrix


# ==============================================================================
# COMPUTE CATEGORY-LEVEL CORRELATIONS (LONG-FORM OUTPUT)
# ==============================================================================
# Complementary to correlation tables - outputs one row per (prompt, category, metric)
# Useful for:
# - Filtering to specific categories or metrics
# - Identifying category-specific patterns (e.g., "chairs easy to find in plausible scenes")
# - Exporting to other analysis tools
# ==============================================================================
def compute_category_correlations(df, prompt_name):
    """Compute per-category correlations for all metrics.
    
    Difference from compute_correlation_table():
    - correlation_table: Wide format (categories × metrics), Excel output
    - category_correlations: Long format (one row per combo), CSV output
    
    Why both formats?
    - Table: Easy to read, compare metrics across categories
    - Long-form: Easy to filter, merge, analyze programmatically
    
    Returns:
        DataFrame with columns: prompt, category, metric, r, p, n
        Each row represents one (category, metric) correlation with nfix
        Returns None if insufficient data
    """
    # Filter to images with nfix data
    df_with_nfix = df[df['nfix'].notna()].copy()
    
    if len(df_with_nfix) == 0:
        print(f"  Warning: No images with nfix for {prompt_name}")
        return None
    
    results = []
    
    # All metrics to analyze (detection, presence, quality, pixel, alexnet)
    all_metrics = (DETECTION_METRICS + PRESENCE_METRICS + QUALITY_METRICS + 
                   PIXEL_METRICS + ALEXNET_METRICS)
    
    for category in CATEGORIES:
        cat_df = df_with_nfix[df_with_nfix['category'] == category]
        
        # Need at least 5 samples for meaningful correlation
        if len(cat_df) < 5:
            continue
        
        for metric in all_metrics:
            if metric not in cat_df.columns:
                continue
            
            # Skip if metric has no variation (check for near-zero std)
            metric_std = cat_df[metric].std()
            nfix_std = cat_df['nfix'].std()
            if np.isclose(metric_std, 0, atol=1e-10) or np.isclose(nfix_std, 0, atol=1e-10):
                continue
            
            # Compute Pearson correlation
            try:
                r, p = stats.pearsonr(cat_df[metric], cat_df['nfix'])
            except:
                continue
            
            results.append({
                'prompt': prompt_name,
                'category': category,
                'metric': metric,
                'r': r,
                'p': p,
                'n': len(cat_df)
            })
    
    if not results:
        return None
    
    return pd.DataFrame(results)


# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================
# Generate publication-ready figures comparing prompts
#
# Plot 1: Quality Comparison (2×2 grid)
#   - Top-left: R-CNN confidence (are objects detected?)
#   - Top-right: YOLO confidence (validation of R-CNN)
#   - Bottom-left: LaRE (is output natural-looking? lower = better)
#   - Bottom-right: Jenga Score (overall preservation, higher = better)
#
# Plot 2: Inter-Prompt Correlation Heatmap
#   - Shows which prompts behave similarly
#   - Color: blue (negative) → white (0) → red (positive)
#   - Use to identify redundant or unique prompts
#
# Plot 3: Metric Correlation Strength
#   - Mean absolute correlation across all prompts
#   - High bars = robust metrics that consistently predict nfix
#   - Use to identify which metrics are most valuable
#
# Plot 4: Hierarchical Clustering Dendrogram
#   - Shows which prompts cluster together based on similarity
#   - Height = dissimilarity (lower = more similar)
#
# Plot 5: Prompt Similarity Heatmap
#   - Enhanced version of inter-prompt correlation with better colormap
#   - Shows dissimilarity (1 - correlation) for easier interpretation
#
# Plot 6: Metric Distribution Comparison
#   - Violin plots comparing key metrics across all prompts
#   - Shows which prompts have different distributions
# ==============================================================================
def create_visualizations(summary_bbox_df, summary_seg_df, inter_prompt_corr, all_correlations):
    """Generate comparison plots for publication/presentation.
    
    Why these visualizations?
    
    1. Quality Comparison:
       - Quick visual ranking of prompts
       - Identifies clear winners/losers
       - Shows trade-offs (e.g., high detection but high LaRE)
    
    2. Inter-Prompt Correlation:
       - Reveals prompt families and outliers
       - Guides prompt selection for experiments
       - Shows which prompts add unique value
    
    3. Metric Strength:
       - Identifies most predictive metrics
       - Helps prioritize which metrics to report in papers
       - Shows robustness across different prompts
    """
    print("\nCreating visualizations...")
    
    # 1. Quality comparison (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if "rcnn_confidence_mean_confidence" in summary_bbox_df.columns:
        ax = axes[0, 0]
        summary_bbox_df.sort_values("rcnn_confidence_mean_confidence", ascending=False).plot(
            x="prompt", y="rcnn_confidence_mean_confidence", kind="bar", ax=ax, legend=False
        )
        ax.set_title("R-CNN Mean Confidence (bbox)")
        ax.set_xlabel("")
        ax.set_ylabel("Confidence")
        ax.tick_params(axis='x', rotation=45)
    
    if "yolo_confidence_mean_confidence" in summary_bbox_df.columns:
        ax = axes[0, 1]
        summary_bbox_df.sort_values("yolo_confidence_mean_confidence", ascending=False).plot(
            x="prompt", y="yolo_confidence_mean_confidence", kind="bar", ax=ax, legend=False, color="orange"
        )
        ax.set_title("YOLO Mean Confidence (bbox)")
        ax.set_xlabel("")
        ax.set_ylabel("Confidence")
        ax.tick_params(axis='x', rotation=45)
    
    if "lare_mean" in summary_bbox_df.columns:
        ax = axes[1, 0]
        summary_bbox_df.sort_values("lare_mean", ascending=True).plot(
            x="prompt", y="lare_mean", kind="bar", ax=ax, legend=False, color="green"
        )
        ax.set_title("LaRE - Lower = Better (bbox)")
        ax.set_xlabel("")
        ax.set_ylabel("LaRE")
        ax.tick_params(axis='x', rotation=45)
    
    if "jenga_score_mean" in summary_bbox_df.columns:
        ax = axes[1, 1]
        summary_bbox_df.sort_values("jenga_score_mean", ascending=False).plot(
            x="prompt", y="jenga_score_mean", kind="bar", ax=ax, legend=False, color="purple"
        )
        ax.set_title("Jenga Score - Higher = Better (bbox)")
        ax.set_xlabel("")
        ax.set_ylabel("Jenga Score")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "01_quality_comparison.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: 01_quality_comparison.png")
    plt.close()
    
    # 2. Inter-prompt correlation heatmap
    if inter_prompt_corr is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(inter_prompt_corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True, ax=ax)
        ax.set_title("Inter-Prompt Correlation\n(Based on metric-nfix relationship patterns)")
        plt.tight_layout()
        plt.savefig(OUTPUT_ROOT / "02_inter_prompt_correlation.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: 02_inter_prompt_correlation.png")
        plt.close()
    
    # 3. Metric correlation strength across all prompts
    if all_correlations:
        # Collect all aggregated correlations
        all_agg_data = []
        for prompt, corr_data in all_correlations.items():
            if corr_data is None:
                continue
            agg_row = corr_data.iloc[-1].to_dict()
            for metric, r_val in agg_row.items():
                if pd.notna(r_val):
                    all_agg_data.append({"prompt": prompt, "metric": metric, "r": abs(r_val)})
        
        if all_agg_data:
            agg_df = pd.DataFrame(all_agg_data)
            metric_strength = agg_df.groupby("metric")["r"].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            metric_strength.plot(kind="bar", ax=ax, color="teal")
            ax.set_title("Mean Absolute Correlation with nfix (Across All Prompts)")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Mean |r|")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUTPUT_ROOT / "03_metric_correlation_strength.png", dpi=300, bbox_inches="tight")
            print(f"  ✓ Saved: 03_metric_correlation_strength.png")
            plt.close()
    
    # 4. Hierarchical Clustering Dendrogram
    if inter_prompt_corr is not None and len(inter_prompt_corr) > 2:
        # Convert correlation to distance
        dist_matrix = 1 - inter_prompt_corr.values
        np.fill_diagonal(dist_matrix, 0)
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = hierarchy.linkage(condensed_dist, method='average')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        hierarchy.dendrogram(linkage_matrix, labels=inter_prompt_corr.index.tolist(), ax=ax,
                            color_threshold=0.005, above_threshold_color='gray')
        ax.set_xlabel('Prompt', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (1 - correlation)', fontsize=12, fontweight='bold')
        ax.set_title('Hierarchical Clustering of Prompts\n(Similar prompts cluster together)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_ROOT / "04_prompt_clustering_dendrogram.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 04_prompt_clustering_dendrogram.png")
        plt.close()
    
    # 5. Enhanced Prompt Similarity Heatmap
    if inter_prompt_corr is not None:
        dissimilarity = 1 - inter_prompt_corr.values
        np.fill_diagonal(dissimilarity, 0)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(dissimilarity * 100, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=inter_prompt_corr.columns, yticklabels=inter_prompt_corr.index,
                    cbar_kws={'label': 'Dissimilarity (%)'}, vmin=0, vmax=1, ax=ax, square=True)
        ax.set_title('Prompt Dissimilarity Matrix\n(0% = identical patterns, higher = more different)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_ROOT / "05_prompt_dissimilarity_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 05_prompt_dissimilarity_heatmap.png")
        plt.close()


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================
# Workflow:
# 1. Load nfix ground truth from COCO-Search18 eye-tracking data
# 2. For each of 12 prompts:
#    a. Load per-image metrics (averaged across 10 reps)
#    b. Compute summary statistics (detection, quality, etc.)
#    c. Compute correlation tables (categories × metrics)
#    d. Save per-prompt Excel file
# 3. Compute inter-prompt correlations (are prompts redundant?)
# 4. Generate visualizations (quality comparison, heatmaps, etc.)
# 5. Print top performers and strongest correlations
#
# Output Files:
# - per_prompt_correlations/{prompt}_nfix_correlations.xlsx (12 files)
#   * bbox tab: category × metric correlation table
#   * segmentation tab: category × metric correlation table
# - summary_statistics.xlsx (bbox + segmentation tabs)
# - summary_statistics_bbox.csv
# - summary_statistics_segmentation.csv
# - category_nfix_correlations.csv (long-form: one row per prompt-category-metric)
# - inter_prompt_correlation_bbox.csv
# - inter_prompt_correlation_segmentation.csv
# - 01_quality_comparison.png
# - 02_inter_prompt_correlation.png
# - 03_metric_correlation_strength.png
# ==============================================================================
def main():
    print("=" * 80)
    print("COMPREHENSIVE PROMPT COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load nfix data
    nfix_map = build_nfix_map(NFIX_JSON)
    
    # Process each prompt
    summary_stats_bbox = []
    summary_stats_segmentation = []
    all_correlations_bbox = {}  # Store correlation tables for inter-prompt analysis
    all_correlations_segmentation = {}
    all_category_corr = []  # Store category-level correlations (long-form)
    
    for prompt in PROMPTS:
        prompt_dir = METRICS_ROOT / prompt
        
        if not prompt_dir.exists():
            print(f"\n⚠️ Skipping {prompt} (directory not found)")
            continue
        
        # Load data
        df = load_per_image_dir(prompt_dir, nfix_map)
        
        if len(df) == 0:
            print(f"  ⚠️ No data loaded for {prompt}")
            continue
        
        # Compute summary statistics
        summary_bbox = compute_summary_statistics(df, prompt, "bbox")
        summary_segmentation = compute_summary_statistics(df, prompt, "segmentation")
        
        if summary_bbox:
            summary_stats_bbox.append(summary_bbox)
        if summary_segmentation:
            summary_stats_segmentation.append(summary_segmentation)
        
        # Compute correlation tables
        print(f"\n  Computing correlation tables for {prompt}...")
        
        bbox_table = compute_correlation_table(df, "bbox", CATEGORIES)
        segmentation_table = compute_correlation_table(df, "segmentation", CATEGORIES)
        
        # Store for inter-prompt analysis
        all_correlations_bbox[prompt] = bbox_table
        all_correlations_segmentation[prompt] = segmentation_table
        
        # Compute category-level correlations (long-form output)
        print(f"  Computing category-level correlations for {prompt}...")
        category_corr = compute_category_correlations(df, prompt)
        if category_corr is not None:
            all_category_corr.append(category_corr)
            print(f"    ✓ {len(category_corr)} category-metric combinations")
        
        # Save per-prompt Excel file
        excel_dir = OUTPUT_ROOT / "per_prompt_correlations"
        excel_dir.mkdir(exist_ok=True)
        excel_path = excel_dir / f"{prompt}_nfix_correlations.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if bbox_table is not None:
                bbox_table.to_excel(writer, sheet_name='bbox')
                print(f"    ✓ bbox correlation table: {bbox_table.shape}")
            if segmentation_table is not None:
                segmentation_table.to_excel(writer, sheet_name='segmentation')
                print(f"    ✓ segmentation correlation table: {segmentation_table.shape}")
        
        print(f"    ✓ Saved: {excel_path.name}")
    
    # Save summary statistics
    print("\n" + "=" * 80)
    print("SAVING SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_bbox_df = pd.DataFrame(summary_stats_bbox)
    summary_segmentation_df = pd.DataFrame(summary_stats_segmentation)
    
    # Excel with tabs
    excel_path = OUTPUT_ROOT / "summary_statistics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_bbox_df.to_excel(writer, sheet_name='bbox', index=False)
        summary_segmentation_df.to_excel(writer, sheet_name='segmentation', index=False)
    print(f"✓ Saved: {excel_path}")
    
    # CSVs
    summary_bbox_df.to_csv(OUTPUT_ROOT / "summary_statistics_bbox.csv", index=False)
    summary_segmentation_df.to_csv(OUTPUT_ROOT / "summary_statistics_segmentation.csv", index=False)
    print(f"✓ Saved: summary_statistics_bbox.csv")
    print(f"✓ Saved: summary_statistics_segmentation.csv")
    
    # Save category-level correlations (long-form)
    if all_category_corr:
        category_corr_df = pd.concat(all_category_corr, ignore_index=True)
        category_corr_df.to_csv(OUTPUT_ROOT / "category_nfix_correlations.csv", index=False)
        print(f"✓ Saved: category_nfix_correlations.csv ({len(category_corr_df)} rows)")
    
    # Compute inter-prompt correlation
    print("\n" + "=" * 80)
    print("COMPUTING INTER-PROMPT CORRELATIONS")
    print("=" * 80)
    
    inter_prompt_corr_bbox = compute_inter_prompt_correlation(all_correlations_bbox)
    inter_prompt_corr_seg = compute_inter_prompt_correlation(all_correlations_segmentation)
    
    if inter_prompt_corr_bbox is not None:
        inter_prompt_corr_bbox.to_csv(OUTPUT_ROOT / "inter_prompt_correlation_bbox.csv")
        print(f"✓ Saved: inter_prompt_correlation_bbox.csv")
    
    if inter_prompt_corr_seg is not None:
        inter_prompt_corr_seg.to_csv(OUTPUT_ROOT / "inter_prompt_correlation_segmentation.csv")
        print(f"✓ Saved: inter_prompt_correlation_segmentation.csv")
    
    # Create visualizations
    create_visualizations(summary_bbox_df, summary_segmentation_df, 
                         inter_prompt_corr_bbox, all_correlations_bbox)
    
    # Print top performers
    print("\n" + "=" * 80)
    print("TOP PERFORMERS - BBOX")
    print("=" * 80)
    print_top_performers(summary_bbox_df)
    
    print("\n" + "=" * 80)
    print("TOP PERFORMERS - SEGMENTATION")
    print("=" * 80)
    print_top_performers(summary_segmentation_df)
    
    # Print strongest correlations
    print("\n" + "=" * 80)
    print("STRONGEST METRIC-NFIX CORRELATIONS (BBOX)")
    print("=" * 80)
    print_strongest_correlations(all_correlations_bbox)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("=" * 80)


def print_strongest_correlations(all_correlations):
    """Print strongest correlations across all prompts.
    
    This identifies:
    - Which prompt+metric combinations best predict human behavior
    - Metrics that are robust across different inpainting strategies
    - Surprising correlations (e.g., unexpected metrics that work well)
    
    Use case: Deciding which metrics to include in your paper's main results
    """
    all_corr_data = []
    
    for prompt, corr_table in all_correlations.items():
        if corr_table is None:
            continue
        
        # Get aggregated correlation row (last row)
        agg_row = corr_table.iloc[-1]
        
        for metric, r_val in agg_row.items():
            if pd.notna(r_val):
                all_corr_data.append({
                    "prompt": prompt,
                    "metric": metric,
                    "r": r_val,
                    "abs_r": abs(r_val)
                })
    
    if not all_corr_data:
        print("  No correlation data available")
        return
    
    df = pd.DataFrame(all_corr_data)
    top = df.nlargest(10, "abs_r")
    
    print("\nTop 10 Strongest Correlations (by absolute value):")
    for _, row in top.iterrows():
        print(f"  {row['prompt']:20s} | {row['metric']:30s} | r={row['r']:+.3f}")


def print_top_performers(summary_df):
    """Print top 3 prompts for each metric.
    
    This quickly shows:
    - Which prompts generate the most recognizable objects (R-CNN)
    - Which prompts produce the most natural results (LaRE)
    - Which prompts best preserve semantic content (DINO)
    - Which prompts excel overall (Jenga Score)
    
    Use case: Executive summary for stakeholders or paper abstract
    """
    if len(summary_df) == 0:
        print("  No data available")
        return
    
    if "rcnn_confidence_mean_confidence" in summary_df.columns:
        print("\nBest R-CNN Detection:")
        top = summary_df.nlargest(3, "rcnn_confidence_mean_confidence")[["prompt", "rcnn_confidence_mean_confidence"]]
        for _, row in top.iterrows():
            print(f"  {row['prompt']}: {row['rcnn_confidence_mean_confidence']:.3f}")
    
    if "lare_mean" in summary_df.columns:
        print("\nBest Quality (Lowest LaRE):")
        top = summary_df.nsmallest(3, "lare_mean")[["prompt", "lare_mean"]]
        for _, row in top.iterrows():
            print(f"  {row['prompt']}: {row['lare_mean']:.1f}")
    
    if "dino_cosine_mean" in summary_df.columns:
        print("\nBest Semantic Preservation (Highest DINO):")
        top = summary_df.nlargest(3, "dino_cosine_mean")[["prompt", "dino_cosine_mean"]]
        for _, row in top.iterrows():
            print(f"  {row['prompt']}: {row['dino_cosine_mean']:.4f}")
    
    if "jenga_score_mean" in summary_df.columns:
        print("\nBest Overall Preservation (Highest Jenga Score):")
        top = summary_df.nlargest(3, "jenga_score_mean")[["prompt", "jenga_score_mean"]]
        for _, row in top.iterrows():
            print(f"  {row['prompt']}: {row['jenga_score_mean']:.4f}")


if __name__ == "__main__":
    main()
