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
- Correlation tables (prompt vs nfix)
- Inter-prompt correlation matrix
- Visualization plots
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Paths
METRICS_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS")
NFIX_JSON = Path("/home/kshaltiel/code/CSE-495-Code/cluster_test_embed/coco_search18_fixations_TP_train_split1.json")
OUTPUT_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_ANALYSIS")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# All prompts to analyze
PROMPTS = [
    "minimal", "contextual", "realistic", "natural_setting", 
    "photorealistic", "original_quality", "plausible", "plausible_scene",
    "plausible_realistic", "plausible_setting", "plausible_placement", 
    "highly_plausible"
]

CATEGORIES = [
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard",
    "knife", "laptop", "microwave", "mouse", "oven", "potted plant",
    "sink", "stop sign", "toilet", "tv"
]

# Metric groups
DETECTION_METRICS = ["rcnn_confidence", "yolo_confidence"]
PRESENCE_METRICS = ["clip_presence_prob", "clip_presence_logit_diff", "clip_text_similarity"]

# Quality/Preservation Metrics:
# - LaRE (Latent Reconstruction Error): How easily a diffusion model can reconstruct the image
#   Lower = more natural/realistic (model finds it easy to denoise)
#   Measures: Does this look like a plausible image that could exist in nature?
#
# - DINO cosine: Semantic similarity in self-supervised vision embedding space
#   Higher = better preservation of semantic content (object shape, scene structure)
#   Measures: Does the inpainted object have similar semantic features to the original?
#
# - CLIP cosine: Perceptual similarity in vision-language embedding space
#   Higher = images "look more similar" from human perception standpoint
#   Measures: Would a human perceive these as similar images?
#
# - Jenga Score: Combined preservation metric = mean_clip_cosine × mean_dino_cosine
#   Higher = both perceptual AND semantic similarity are high (robust preservation)
#   Multiplication ensures BOTH metrics must agree (not just one)
#   Measures: Overall consistency and quality of preservation
#
# Note: These aren't "image quality" in the traditional sense (sharpness, noise, artifacts)
# but rather "semantic/perceptual preservation" - did inpainting maintain the right visual concepts?
QUALITY_METRICS = ["lare", "dino_cosine", "clip_cosine", "jenga_score"]

PIXEL_METRICS = ["pixel_l2", "pixel_cosine"]
ALEXNET_METRICS = ["alex_low_l2", "alex_low_cosine", "alex_mid_l2", "alex_mid_cosine", 
                   "alex_high_l2", "alex_high_cosine"]


# ==============================================================================
# LOAD NFIX DATA (HUMAN VISUAL SEARCH DIFFICULTY)
# ==============================================================================
# nfix = number of fixations before finding target object
# Lower nfix = easier to find (visually salient, distinct)
# Higher nfix = harder to find (blends in, complex scene)
#
# This is our ground truth for "how humans perceive these images"
# Strong metric-nfix correlations mean our metrics predict human behavior
# ==============================================================================
def load_nfix_data():
    """
    Load nfix (number of fixations) data for all images.
    
    What is nfix?
    - Human eye-tracking metric: how many fixations before finding target object
    - Lower nfix = object is easy to find (salient, distinct, simple scene)
    - Higher nfix = object is hard to find (camouflaged, complex scene, small)
    
    Why do we care?
    - nfix represents human visual perception difficulty
    - If our metrics correlate with nfix, it means they predict human behavior
    - Good inpainting should preserve nfix relationships (if original was easy to find,
      inpainted version should also be easy to find)
    
    Returns:
    - Dictionary mapping (category, task_id) -> average nfix value
    """
    print("Loading nfix data...")
    with open(NFIX_JSON) as f:
        nfix_data = json.load(f)
    
    # Build map: (category, task_id) -> nfix value
    nfix_map = {}
    for entry in nfix_data:
        cat = entry.get("category")
        task_id = entry.get("task_id")
        nfix = entry.get("nfix")
        if cat and task_id is not None and nfix is not None:
            key = (cat, int(task_id))
            if key not in nfix_map:
                nfix_map[key] = []
            nfix_map[key].append(nfix)
    
    # Average multiple nfix values per image
    nfix_avg = {k: np.mean(v) for k, v in nfix_map.items()}
    print(f"  Loaded nfix for {len(nfix_avg)} unique images")
    return nfix_avg


# ==============================================================================
# LOAD METRICS FOR A SINGLE PROMPT
# ==============================================================================
# Each prompt (minimal, plausible, realistic, etc.) has a directory of JSON files
# Each JSON contains metrics for one image (averaged across 10 repetitions)
#
# We load all images for a prompt and attach nfix values where available
# This creates a dataset we can analyze to see prompt performance
# ==============================================================================
def load_prompt_metrics(prompt_name, nfix_map):
    """
    Load all metrics for a given prompt strategy.
    
    Process:
    1. Find all JSON files in PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/{prompt}/
    2. Parse filename to extract category and task_id
    3. Load metric arrays and average across repetitions
    4. Attach nfix value if available for this image
    5. Return DataFrame with one row per image
    
    Why average across repetitions?
    - Each image was inpainted 10 times (rep0-rep9) to account for randomness
    - We take mean to get stable metric estimates
    
    Returns:
    - DataFrame with columns: prompt, category, task_id, nfix, [all metrics]
    """
    print(f"\nLoading metrics for: {prompt_name}")
    prompt_dir = METRICS_ROOT / prompt_name
    
    if not prompt_dir.exists():
        print(f"  ⚠️ Directory not found: {prompt_dir}")
        return None
    
    all_data = []
    
    for json_file in prompt_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: category_taskid_bbox.json or category_taskid_segmentation.json
            stem = json_file.stem
            parts = stem.rsplit("_", 2)
            if len(parts) < 3:
                continue
            
            category = parts[0]
            task_id_str = parts[1]
            mask_type = parts[2]  # "bbox" or "segmentation"
            
            # Only process bbox or segmentation files
            if mask_type not in ["bbox", "segmentation"]:
                continue
            
            try:
                task_id = int(task_id_str)
            except ValueError:
                continue
            
            # Get nfix if available
            nfix_key = (category, task_id)
            nfix = nfix_map.get(nfix_key)
            
            # Extract metrics (average across reps)
            row = {
                "prompt": prompt_name,
                "category": category,
                "task_id": task_id,
                "mask_type": mask_type,  # "bbox" or "segmentation"
                "nfix": nfix
            }
            
            # Average all metric arrays
            for metric_key, metric_values in data.items():
                if isinstance(metric_values, list) and len(metric_values) > 0:
                    row[metric_key] = np.mean(metric_values)
            
            # Compute jenga_score: mean_clip_cosine × mean_dino_cosine
            # Higher = better preservation (both perceptual and semantic similarity agree)
            if "clip_cosine" in row and "dino_cosine" in row:
                row["jenga_score"] = row["clip_cosine"] * row["dino_cosine"]
            
            all_data.append(row)
        
        except Exception as e:
            continue
    
    df = pd.DataFrame(all_data)
    print(f"  Loaded {len(df)} images")
    print(f"  - bbox: {(df['mask_type'] == 'bbox').sum()}")
    print(f"  - segmentation: {(df['mask_type'] == 'segmentation').sum()}")
    print(f"  Images with nfix: {df['nfix'].notna().sum()}")
    
    return df


# ==============================================================================
# COMPUTE SUMMARY STATISTICS FOR A PROMPT
# ==============================================================================
# Aggregate all images for a prompt into single summary metrics
# This gives us a "report card" for each prompt strategy
#
# Key Questions:
# 1. Detection: Did the prompt generate recognizable objects? (R-CNN/YOLO)
# 2. Presence: Did CLIP still detect the target object? (CLIP presence)
# 3. Quality: How natural/realistic are the results? (LaRE, DINO, CLIP)
#
# NEW: Now computed separately for bbox and segmentation masks
# ==============================================================================
def compute_summary_statistics(df, prompt_name, mask_type=None):
    """
    Compute aggregate statistics for a prompt across all images.
    
    Args:
        df: DataFrame with all images for this prompt
        prompt_name: Name of the prompt (e.g., "minimal")
        mask_type: "bbox", "segmentation", or None (for all combined)
    
    Metrics computed:
    
    Detection Success:
    - rcnn/yolo_detection_rate: % images with confidence > 0.5
    - rcnn/yolo_mean_confidence: average detection confidence
    Higher = prompt generates more recognizable objects
    
    Presence Preservation:
    - clip_presence_mean: average CLIP probability that target is present
    - clip_logit_diff_mean: average confidence in presence vs absence
    Higher = target object is still clearly visible after inpainting
    
    Quality/Naturalness:
    - lare_mean/median: average Latent Reconstruction Error
      Lower = more natural/realistic (diffusion model easily reconstructs it)
    - dino_cosine_mean: semantic similarity to original
      Higher = better preservation of object shape/structure
    - clip_cosine_mean: perceptual similarity to original
      Higher = looks more similar from human perception view
    
    Returns:
    - Dictionary with summary statistics for this prompt
    """
    # Filter by mask type if specified
    if mask_type is not None:
        df = df[df["mask_type"] == mask_type].copy()
    
    summary = {"prompt": prompt_name}
    if mask_type is not None:
        summary["mask_type"] = mask_type
    
    # Count
    summary["total_images"] = len(df)
    summary["images_with_nfix"] = df["nfix"].notna().sum()
    
    # Detection success (% with confidence > 0.5)
    if "rcnn_confidence" in df.columns:
        summary["rcnn_detection_rate"] = (df["rcnn_confidence"] > 0.5).mean()
        summary["rcnn_mean_confidence"] = df["rcnn_confidence"].mean()
    
    if "yolo_confidence" in df.columns:
        summary["yolo_detection_rate"] = (df["yolo_confidence"] > 0.5).mean()
        summary["yolo_mean_confidence"] = df["yolo_confidence"].mean()
    
    # Presence preservation
    if "clip_presence_prob" in df.columns:
        summary["clip_presence_mean"] = df["clip_presence_prob"].mean()
        summary["clip_presence_std"] = df["clip_presence_prob"].std()
    
    if "clip_presence_logit_diff" in df.columns:
        summary["clip_logit_diff_mean"] = df["clip_presence_logit_diff"].mean()
    
    # Quality metrics (lower LaRE is better, higher DINO/CLIP cosine is better)
    if "lare" in df.columns:
        summary["lare_mean"] = df["lare"].mean()
        summary["lare_median"] = df["lare"].median()
        summary["lare_std"] = df["lare"].std()
    
    if "dino_cosine" in df.columns:
        summary["dino_cosine_mean"] = df["dino_cosine"].mean()
        summary["dino_cosine_std"] = df["dino_cosine"].std()
    
    if "clip_cosine" in df.columns:
        summary["clip_cosine_mean"] = df["clip_cosine"].mean()
        summary["clip_cosine_std"] = df["clip_cosine"].std()
    
    # Jenga score (combined CLIP × DINO preservation)
    if "jenga_score" in df.columns:
        summary["jenga_score_mean"] = df["jenga_score"].mean()
        summary["jenga_score_std"] = df["jenga_score"].std()
    
    # Pixel metrics
    if "pixel_l2" in df.columns:
        summary["pixel_l2_mean"] = df["pixel_l2"].mean()
    
    if "pixel_cosine" in df.columns:
        summary["pixel_cosine_mean"] = df["pixel_cosine"].mean()
    
    return summary


# ==============================================================================
# COMPUTE CORRELATIONS BETWEEN METRICS AND NFIX
# ==============================================================================
# This is the core analysis: Do our metrics predict human perception?
#
# We compute Pearson and Spearman correlations between each metric and nfix
# - Pearson: Linear relationship (e.g., as metric increases, nfix increases)
# - Spearman: Monotonic relationship (rank-based, handles non-linearity)
#
# Strong correlations (|r| > 0.3) mean the metric predicts visual search difficulty
# Example: If alex_high_l2 correlates +0.5 with nfix, it means:
#   "Images with high L2 distance are harder to find" - the metric captures salience
#
# We want prompts that preserve these correlations from the original images
# ==============================================================================
def compute_nfix_correlations(df, prompt_name, mask_type=None):
    """
    Compute correlations between all metrics and nfix for this prompt.
    
    Why correlate with nfix?
    - nfix measures human visual search difficulty (ground truth)
    - If a metric correlates with nfix, it predicts human behavior
    - Good prompts should preserve the correlation patterns seen in originals
    
    Example interpretation:
    - pixel_l2 vs nfix: r=+0.4 means "higher pixel difference = harder to find"
      This makes sense: different-looking objects are more salient/easier
    - lare vs nfix: r=+0.3 means "less natural images = harder to find"
      Natural-looking objects might blend in more
    
    Args:
        df: DataFrame with metrics for this prompt
        prompt_name: Name of the prompt
        mask_type: "bbox", "segmentation", or None (for all combined)
    
    Returns:
    - DataFrame with columns: prompt, mask_type, metric, r_pearson, p_pearson, r_spearman, p_spearman
    - Each row is one metric's correlation with nfix
    """
    # Filter by mask type if specified
    if mask_type is not None:
        df_with_nfix = df[(df["nfix"].notna()) & (df["mask_type"] == mask_type)].copy()
    else:
        df_with_nfix = df[df["nfix"].notna()].copy()
    
    if len(df_with_nfix) < 10:
        print(f"  ⚠️ Not enough data with nfix for {prompt_name}" + (f" ({mask_type})" if mask_type else ""))
        return None
    
    results = []
    
    # All metrics to correlate
    all_metrics = (DETECTION_METRICS + PRESENCE_METRICS + QUALITY_METRICS + 
                   PIXEL_METRICS + ALEXNET_METRICS)
    
    for metric in all_metrics:
        if metric not in df_with_nfix.columns:
            continue
        
        # Remove NaN values
        valid_mask = df_with_nfix[metric].notna()
        if valid_mask.sum() < 10:
            continue
        
        x = df_with_nfix.loc[valid_mask, "nfix"].values
        y = df_with_nfix.loc[valid_mask, metric].values
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(x, y)
        
        # Spearman correlation
        r_spearman, p_spearman = stats.spearmanr(x, y)
        
        result = {
            "prompt": prompt_name,
            "metric": metric,
            "r_pearson": r_pearson,
            "p_pearson": p_pearson,
            "r_spearman": r_spearman,
            "p_spearman": p_spearman,
            "n_samples": len(x)
        }
        if mask_type is not None:
            result["mask_type"] = mask_type
        results.append(result)
    
    return pd.DataFrame(results)


# ==============================================================================
# COMPUTE PER-CATEGORY CORRELATIONS
# ==============================================================================
# Some prompts might work better for certain object types
# e.g., "plausible_setting" might excel at furniture but fail at vehicles
#
# By computing correlations separately per category, we can identify:
# - Which prompts are consistent across all objects
# - Which prompts specialize in specific categories
# - Whether certain objects are inherently harder to inpaint well
# ==============================================================================
def compute_category_correlations(df, prompt_name):
    """
    Compute nfix correlations separately for each category (bottle, chair, etc.).
    
    Why per-category analysis?
    - Different object types have different visual characteristics
    - A prompt might excel at inpainting furniture but fail at small objects
    - Category-level correlations reveal specialization patterns
    
    Example insights:
    - If "realistic" has strong correlations for all categories: consistent prompt
    - If "plausible_setting" only works for furniture: specialized prompt
    - If "bottle" has weak correlations across all prompts: inherently hard object
    
    Returns:
    - DataFrame with columns: prompt, category, metric, r, p, n
    - Each row is one (prompt, category, metric) combination
    """
    df_with_nfix = df[df["nfix"].notna()].copy()
    
    results = []
    
    for category in CATEGORIES:
        cat_df = df_with_nfix[df_with_nfix["category"] == category]
        
        if len(cat_df) < 5:
            continue
        
        for metric in DETECTION_METRICS + QUALITY_METRICS:
            if metric not in cat_df.columns:
                continue
            
            valid_mask = cat_df[metric].notna()
            if valid_mask.sum() < 5:
                continue
            
            x = cat_df.loc[valid_mask, "nfix"].values
            y = cat_df.loc[valid_mask, metric].values
            
            r, p = stats.pearsonr(x, y)
            
            results.append({
                "prompt": prompt_name,
                "category": category,
                "metric": metric,
                "r": r,
                "p": p,
                "n": len(x)
            })
    
    return pd.DataFrame(results)


# ==============================================================================
# INTER-PROMPT CORRELATION (META-ANALYSIS)
# ==============================================================================
# This answers: "Are different prompts actually producing different results?"
#
# Method:
# 1. For each prompt, we have a vector of (metric -> r_pearson with nfix)
# 2. Correlate these vectors between prompts
# 3. High inter-prompt correlation = prompts behave similarly
#
# Example:
# - If "minimal" and "contextual" have inter-prompt r=0.95, they're nearly identical
# - If "plausible" and "realistic" have r=0.40, they're quite different approaches
#
# This helps us identify:
# - Redundant prompts (can drop similar ones)
# - Unique prompts (keep these for diversity)
# - Clusters of similar strategies
# ==============================================================================
def compute_inter_prompt_correlation(all_correlations_df):
    """
    Compute how similar different prompts are based on their correlation patterns.
    
    What does this measure?
    - NOT: "Do prompts produce visually similar images?"
    - YES: "Do prompts have similar metric-nfix relationship patterns?"
    
    How it works:
    1. Each prompt has a "signature": vector of correlations for all metrics
       Example: minimal = [alex_l2: +0.4, dino: -0.2, lare: +0.3, ...]
    2. We correlate these signature vectors between prompts
    3. High correlation = prompts produce similar perceptual patterns
    
    Use cases:
    - Identify redundant prompts (r > 0.9): picking one is sufficient
    - Find complementary prompts (r < 0.5): both provide unique insights
    - Cluster prompts into families (e.g., all "plausible" variants)
    
    Returns:
    - Correlation matrix (DataFrame): prompts × prompts with correlation values
    """
    print("\nComputing inter-prompt correlation matrix...")
    
    # Pivot to get: rows=metrics, columns=prompts, values=r_pearson
    pivot = all_correlations_df.pivot(index="metric", columns="prompt", values="r_pearson")
    
    # Compute correlation between prompt columns
    prompt_corr = pivot.corr(method="pearson")
    
    return prompt_corr


# ==============================================================================
# CREATE VISUALIZATION PLOTS
# ==============================================================================
# Generate publication-ready figures comparing prompts
#
# Plot 1: Quality Comparison (4-panel)
#   - R-CNN detection confidence (higher = better object generation)
#   - YOLO detection confidence (validation of R-CNN)
#   - LaRE (lower = more natural/realistic)
#   - DINO cosine (higher = better semantic preservation)
#
# Plot 2: Inter-Prompt Correlation Heatmap
#   - Shows which prompts produce similar perceptual patterns
#   - Diagonal is always 1.0 (prompt correlates with itself)
#   - Off-diagonal values reveal similarity/difference
#
# Plot 3: Metric Correlation Strength
#   - Which metrics are best at predicting nfix across all prompts?
#   - High bars = robust metrics that work consistently
# ==============================================================================
def create_visualizations(summary_df, all_correlations_df, inter_prompt_corr):
    """
    Create comparison plots for publication/presentation.
    
    Generated plots:
    
    1. 01_quality_comparison.png (2×2 grid):
       - Top-left: R-CNN confidence (are objects detected?)
       - Top-right: YOLO confidence (validation)
       - Bottom-left: LaRE (is output natural-looking?)
       - Bottom-right: DINO cosine (semantic preservation)
       
    2. 02_inter_prompt_correlation.png (heatmap):
       - Shows which prompts behave similarly
       - Color scale: blue (negative) → white (0) → red (positive)
       - Use this to identify redundant or unique prompts
       
    3. 03_metric_correlation_strength.png (bar chart):
       - Mean absolute correlation across all prompts
       - Identifies which metrics are most predictive of nfix
       - High bars = robust metrics worth reporting in papers
    """
    print("\nCreating visualizations...")
    
    # 1. Detection success comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R-CNN detection
    if "rcnn_mean_confidence" in summary_df.columns:
        ax = axes[0, 0]
        summary_df.sort_values("rcnn_mean_confidence", ascending=False).plot(
            x="prompt", y="rcnn_mean_confidence", kind="bar", ax=ax, legend=False
        )
        ax.set_title("R-CNN Mean Confidence")
        ax.set_xlabel("")
        ax.set_ylabel("Confidence")
        ax.tick_params(axis='x', rotation=45)
    
    # YOLO detection
    if "yolo_mean_confidence" in summary_df.columns:
        ax = axes[0, 1]
        summary_df.sort_values("yolo_mean_confidence", ascending=False).plot(
            x="prompt", y="yolo_mean_confidence", kind="bar", ax=ax, legend=False, color="orange"
        )
        ax.set_title("YOLO Mean Confidence")
        ax.set_xlabel("")
        ax.set_ylabel("Confidence")
        ax.tick_params(axis='x', rotation=45)
    
    # LaRE (lower is better)
    if "lare_mean" in summary_df.columns:
        ax = axes[1, 0]
        summary_df.sort_values("lare_mean", ascending=True).plot(
            x="prompt", y="lare_mean", kind="bar", ax=ax, legend=False, color="green"
        )
        ax.set_title("LaRE (Lower = Better Quality)")
        ax.set_xlabel("")
        ax.set_ylabel("LaRE")
        ax.tick_params(axis='x', rotation=45)
    
    # DINO cosine (higher is better)
    if "dino_cosine_mean" in summary_df.columns:
        ax = axes[1, 1]
        summary_df.sort_values("dino_cosine_mean", ascending=False).plot(
            x="prompt", y="dino_cosine_mean", kind="bar", ax=ax, legend=False, color="purple"
        )
        ax.set_title("DINO Cosine (Higher = Better)")
        ax.set_xlabel("")
        ax.set_ylabel("Cosine Similarity")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "01_quality_comparison.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_ROOT / '01_quality_comparison.png'}")
    plt.close()
    
    # 2. Inter-prompt correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(inter_prompt_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Inter-Prompt Correlation\n(Based on metric-nfix relationship patterns)")
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "02_inter_prompt_correlation.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_ROOT / '02_inter_prompt_correlation.png'}")
    plt.close()
    
    # 3. Correlation strength by metric
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get mean absolute correlation for each metric across prompts
    metric_strength = all_correlations_df.groupby("metric")["r_pearson"].apply(
        lambda x: np.abs(x).mean()
    ).sort_values(ascending=False)
    
    metric_strength.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Mean Absolute Correlation with nfix (Across All Prompts)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean |r|")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "03_metric_correlation_strength.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_ROOT / '03_metric_correlation_strength.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("COMPREHENSIVE PROMPT COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load nfix data
    nfix_map = load_nfix_data()
    
    # Load metrics for all prompts
    all_prompt_data = {}
    summary_stats_bbox = []
    summary_stats_segmentation = []
    all_correlations = []
    category_correlations = []
    prompt_correlation_dfs = {}  # Store per-prompt correlation dataframes
    
    for prompt in PROMPTS:
        df = load_prompt_metrics(prompt, nfix_map)
        
        if df is None or len(df) == 0:
            print(f"  ⚠️ Skipping {prompt} (no data)")
            continue
        
        all_prompt_data[prompt] = df
        
        # Compute summary stats separately for bbox and segmentation
        summary_bbox = compute_summary_statistics(df, prompt, mask_type="bbox")
        summary_stats_bbox.append(summary_bbox)
        
        summary_segmentation = compute_summary_statistics(df, prompt, mask_type="segmentation")
        summary_stats_segmentation.append(summary_segmentation)
        
        # Compute nfix correlations separately for bbox and segmentation
        corr_bbox = compute_nfix_correlations(df, prompt, mask_type="bbox")
        corr_segmentation = compute_nfix_correlations(df, prompt, mask_type="segmentation")
        
        # Store per-prompt correlations for individual Excel files
        prompt_correlation_dfs[prompt] = {
            "bbox": corr_bbox,
            "segmentation": corr_segmentation
        }
        
        # Also add to combined list (using bbox for overall analysis)
        if corr_bbox is not None:
            all_correlations.append(corr_bbox)
        
        # Compute category-level correlations
        cat_corr_df = compute_category_correlations(df, prompt)
        if len(cat_corr_df) > 0:
            category_correlations.append(cat_corr_df)
    
    # Create summary dataframes
    summary_bbox_df = pd.DataFrame(summary_stats_bbox)
    summary_segmentation_df = pd.DataFrame(summary_stats_segmentation)
    all_correlations_df = pd.concat(all_correlations, ignore_index=True) if all_correlations else pd.DataFrame()
    category_correlations_df = pd.concat(category_correlations, ignore_index=True) if category_correlations else pd.DataFrame()
    
    # Save summary statistics
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save as Excel with separate tabs for bbox and segmentation
    excel_path = OUTPUT_ROOT / "summary_statistics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_bbox_df.to_excel(writer, sheet_name='bbox', index=False)
        summary_segmentation_df.to_excel(writer, sheet_name='segmentation', index=False)
    print(f"✓ Saved: {excel_path}")
    print(f"  - bbox tab: {len(summary_bbox_df)} prompts")
    print(f"  - segmentation tab: {len(summary_segmentation_df)} prompts")
    
    # Also save as separate CSVs for easy loading
    summary_bbox_df.to_csv(OUTPUT_ROOT / "summary_statistics_bbox.csv", index=False)
    summary_segmentation_df.to_csv(OUTPUT_ROOT / "summary_statistics_segmentation.csv", index=False)
    print(f"✓ Saved: {OUTPUT_ROOT / 'summary_statistics_bbox.csv'}")
    print(f"✓ Saved: {OUTPUT_ROOT / 'summary_statistics_segmentation.csv'}")
    
    all_correlations_df.to_csv(OUTPUT_ROOT / "all_nfix_correlations.csv", index=False)
    print(f"✓ Saved: {OUTPUT_ROOT / 'all_nfix_correlations.csv'}")
    
    category_correlations_df.to_csv(OUTPUT_ROOT / "category_nfix_correlations.csv", index=False)
    print(f"✓ Saved: {OUTPUT_ROOT / 'category_nfix_correlations.csv'}")
    
    # Save per-prompt correlation Excel files
    print(f"\n✓ Saving per-prompt correlation spreadsheets...")
    per_prompt_dir = OUTPUT_ROOT / "per_prompt_correlations"
    per_prompt_dir.mkdir(exist_ok=True)
    
    for prompt, corr_data in prompt_correlation_dfs.items():
        excel_path = per_prompt_dir / f"{prompt}_nfix_correlations.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if corr_data["bbox"] is not None:
                corr_data["bbox"].to_excel(writer, sheet_name='bbox', index=False)
            if corr_data["segmentation"] is not None:
                corr_data["segmentation"].to_excel(writer, sheet_name='segmentation', index=False)
        
        print(f"  - {prompt}_nfix_correlations.xlsx")
    
    # Compute inter-prompt correlation
    if len(all_correlations_df) > 0:
        inter_prompt_corr = compute_inter_prompt_correlation(all_correlations_df)
        inter_prompt_corr.to_csv(OUTPUT_ROOT / "inter_prompt_correlation_matrix.csv")
        print(f"✓ Saved: {OUTPUT_ROOT / 'inter_prompt_correlation_matrix.csv'}")
        
        # Create visualizations (use bbox for primary plots)
        create_visualizations(summary_bbox_df, all_correlations_df, inter_prompt_corr)
    
    # Print top performers (bbox and segmentation)
    print("\n" + "=" * 80)
    print("TOP PERFORMERS - BBOX")
    print("=" * 80)
    print_top_performers(summary_bbox_df)
    
    print("\n" + "=" * 80)
    print("TOP PERFORMERS - SEGMENTED")
    print("=" * 80)
    print_top_performers(summary_segmentation_df)
    
    # Strongest overall correlations
    if len(all_correlations_df) > 0:
        print("\n" + "=" * 80)
        print("STRONGEST METRIC-NFIX CORRELATIONS")
        print("=" * 80)
        all_correlations_df["abs_r"] = all_correlations_df["r_pearson"].abs()
        top_corr = all_correlations_df.nlargest(5, "abs_r")[["prompt", "metric", "r_pearson", "p_pearson"]]
        for _, row in top_corr.iterrows():
            sig = "***" if row["p_pearson"] < 0.001 else "**" if row["p_pearson"] < 0.01 else "*" if row["p_pearson"] < 0.05 else ""
            print(f"  {row['prompt']:20s} | {row['metric']:25s} | r={row['r_pearson']:+.3f} {sig}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("=" * 80)


def print_top_performers(summary_df):
    """Print top 3 prompts for each metric."""
    if "rcnn_mean_confidence" in summary_df.columns:
def print_top_performers(summary_df):
    """Print top 3 prompts for each metric."""
    if "rcnn_mean_confidence" in summary_df.columns:
        print("\nBest R-CNN Detection:")
        top = summary_df.nlargest(3, "rcnn_mean_confidence")[["prompt", "rcnn_mean_confidence"]]
        for _, row in top.iterrows():
            print(f"  {row['prompt']}: {row['rcnn_mean_confidence']:.3f}")
    
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

