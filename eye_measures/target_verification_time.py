"""
target_verification_time.py

Calculate target verification time (last fixation duration) and correlate with image quality metrics.
Uses precomputed metrics from plausible_realistic prompt experiments.
Eye metric: RT - T[first_target_index] (time from first landing on target to response)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
PER_IMAGE_JSON_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/plausible_realistic")
FIXATION_DATA = Path("/home/kshaltiel/cluster_test_embed/coco_search18_fixations_TP_train_split1.json")
BASE_OUTPUT_DIR = Path("/home/kshaltiel/code/CSE-495-Code/metrics_calculations/outputs")
BLUR_TYPES = ["bbox", "segmentation"]

IMAGE_METRICS = [
    "pixel_L2", "pixel_cosine",
    "low_L2", "low_cosine",
    "mid_L2", "mid_cosine",
    "high_L2", "high_cosine",
    "clip_cosine", "clip_L2",
    "dino_cosine", "dino_L2",
    "rcnn_confidence", "yolo_confidence",
    "clip_presence_prob", "clip_presence_logit_diff", "clip_text_similarity"
]


def point_in_bbox(x, y, bbox):
    """Check if point (x, y) is inside bbox [x, y, width, height]."""
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def load_fixation_data():
    """Load fixation data and extract target verification times."""
    with open(FIXATION_DATA, 'r') as f:
        data = json.load(f)
    
    eye_data = defaultdict(lambda: defaultdict(list))
    
    for trial in data:
        # Only target-present, correct trials
        if trial.get('condition') != 'present' or trial.get('correct') != 1:
            continue
        
        T = trial.get('T', [])
        X = trial.get('X', [])
        Y = trial.get('Y', [])
        bbox = trial.get('bbox', [])
        RT = trial.get('RT', 0)
        
        # Need at least one fixation and a valid bbox
        if len(T) == 0 or len(bbox) != 4:
            continue
        
        # Find first fixation on target
        first_target_index = None
        for i in range(len(X)):
            if point_in_bbox(X[i], Y[i], bbox):
                first_target_index = i
                break
        
        if first_target_index is None:
            continue
        
        # T contains durations; cumulative time to landing = sum(T[0:first_target_index])
        # Verification time = time from landing on target to response
        verification_time = RT - sum(T[0:first_target_index])
        
        # Sanity check
        if verification_time < 0:
            verification_time = 0
        
        image_name = trial['name']
        task = trial['task']
        eye_data[task][image_name].append(verification_time)
    
    return eye_data


def load_image_metrics(blur_type):
    """Load precomputed image metrics from JSON files."""
    image_metrics = defaultdict(lambda: defaultdict(dict))
    
    for json_file in PER_IMAGE_JSON_ROOT.glob(f"*_{blur_type}.json"):
        parts = json_file.stem.split('_')
        if len(parts) < 3:
            continue
        
        category = parts[0]
        imageid = parts[1]
        image_name = f"{imageid}.jpg"
        
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        
        # Take mean across repetitions
        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
        image_metrics[category][image_name] = metrics_mean
    
    return image_metrics


def match_data(eye_data, image_metrics):
    """Match eye tracking data with image quality metrics."""
    matched = []
    
    for category in eye_data.keys():
        for image_name in eye_data[category].keys():
            if image_name not in image_metrics[category]:
                continue
            
            # One row per trial (subject) - do NOT pre-aggregate
            eye_values = eye_data[category][image_name]
            for eye_val in eye_values:
                row = {
                    'category': category,
                    'image_name': image_name,
                    'verification_time': eye_val
                }
                row.update(image_metrics[category][image_name])
                matched.append(row)
    
    return pd.DataFrame(matched)


def compute_correlations(df):
    """Compute correlations between verification time and image metrics."""
    correlations = {}
    
    for metric in IMAGE_METRICS:
        if metric not in df.columns:
            continue
        
        valid_mask = df['verification_time'].notna() & df[metric].notna()
        if valid_mask.sum() < 3:
            continue
        
        try:
            r, p = pearsonr(df.loc[valid_mask, 'verification_time'], 
                          df.loc[valid_mask, metric])
            correlations[metric] = {
                'r': r,
                'p': p,
                'n': valid_mask.sum()
            }
        except Exception:
            continue
    
    return pd.DataFrame(correlations).T


def compute_correlations_by_category_with_aggregation(df, stat_func='mean'):
    """Compute correlations per category, aggregating eye metric values per image first.
    
    Args:
        df: DataFrame with columns: category, image_name, verification_time, and image metrics
        stat_func: 'mean', 'median', 'min', 'max', or 'std' - how to aggregate eye metric per image
    
    Returns:
        DataFrame with categories as rows, image metrics as columns (r values)
    """
    categories = sorted(df['category'].unique())
    corr_matrix = pd.DataFrame(index=categories, columns=IMAGE_METRICS, dtype=float)
    
    for category in categories:
        cat_df = df[df['category'] == category].copy()
        
        # Aggregate eye metric per image using the specified statistic
        if stat_func == 'mean':
            agg_eye = cat_df.groupby('image_name')['verification_time'].mean()
        elif stat_func == 'median':
            agg_eye = cat_df.groupby('image_name')['verification_time'].median()
        elif stat_func == 'min':
            agg_eye = cat_df.groupby('image_name')['verification_time'].min()
        elif stat_func == 'max':
            agg_eye = cat_df.groupby('image_name')['verification_time'].max()
        elif stat_func == 'std':
            agg_eye = cat_df.groupby('image_name')['verification_time'].std()
        else:
            continue
        
        # Get image metrics
        img_metrics_df = cat_df.groupby('image_name')[IMAGE_METRICS].first()
        
        # Align indices
        common_idx = agg_eye.index.intersection(img_metrics_df.index)
        if len(common_idx) < 3:
            continue
        
        agg_eye_aligned = agg_eye[common_idx]
        
        # Correlate aggregated eye metric with each image metric
        for metric in IMAGE_METRICS:
            if metric not in img_metrics_df.columns:
                continue
            
            img_values = img_metrics_df.loc[common_idx, metric]
            valid_mask = agg_eye_aligned.notna() & img_values.notna()
            
            if valid_mask.sum() < 3:
                continue
            
            try:
                r, p = pearsonr(agg_eye_aligned[valid_mask], img_values[valid_mask])
                corr_matrix.at[category, metric] = r
            except Exception:
                continue
    
    return corr_matrix


def compute_correlations_by_category(df):
    """Compute correlations separately for each category.
    Returns DataFrame with categories as rows and metrics as columns (r values only)."""
    categories = sorted(df['category'].unique())
    corr_matrix = pd.DataFrame(index=categories, columns=IMAGE_METRICS, dtype=float)
    
    for category in categories:
        cat_df = df[df['category'] == category]
        if len(cat_df) < 3:
            continue
        
        for metric in IMAGE_METRICS:
            if metric not in cat_df.columns:
                continue
            
            valid_mask = cat_df['verification_time'].notna() & cat_df[metric].notna()
            if valid_mask.sum() < 3:
                continue
            
            try:
                r, p = pearsonr(cat_df.loc[valid_mask, 'verification_time'],
                              cat_df.loc[valid_mask, metric])
                corr_matrix.at[category, metric] = r
            except Exception:
                continue
    
    return corr_matrix


def compute_statistics_by_category(df):
    """Compute statistics for eye metric and image metrics grouped by category."""
    categories = sorted(df['category'].unique())
    all_metrics = ['verification_time'] + IMAGE_METRICS
    
    stats_dfs = {}
    for stat in ['mean', 'median', 'min', 'max', 'std']:
        stats_df = pd.DataFrame(index=categories, columns=all_metrics, dtype=float)
        
        for category in categories:
            cat_data = df[df['category'] == category]
            for metric in all_metrics:
                if metric not in cat_data.columns:
                    continue
                values = cat_data[metric].dropna()
                if len(values) == 0:
                    continue
                
                if stat == 'mean':
                    stats_df.at[category, metric] = values.mean()
                elif stat == 'median':
                    stats_df.at[category, metric] = values.median()
                elif stat == 'min':
                    stats_df.at[category, metric] = values.min()
                elif stat == 'max':
                    stats_df.at[category, metric] = values.max()
                elif stat == 'std':
                    stats_df.at[category, metric] = values.std()
        
        stats_dfs[stat] = stats_df
    
    return stats_dfs


def correlate_statistics_across_categories(stats_dfs, eye_metric_name='verification_time'):
    """Correlate statistics of eye metric with statistics of image metrics across categories.
    
    For each statistic (mean, median, etc.), correlate the eye metric statistic with each 
    image metric statistic across all categories.
    
    Returns: DataFrame with statistics as rows, image metrics as columns.
    """
    stat_names = ['mean', 'median', 'min', 'max', 'std']
    corr_results = pd.DataFrame(index=stat_names, columns=IMAGE_METRICS, dtype=float)
    
    for stat in stat_names:
        stats_df = stats_dfs[stat]
        eye_values = stats_df[eye_metric_name].dropna()
        
        for img_metric in IMAGE_METRICS:
            if img_metric not in stats_df.columns:
                continue
            
            img_values = stats_df[img_metric].dropna()
            
            # Align indices
            common_idx = eye_values.index.intersection(img_values.index)
            if len(common_idx) < 3:
                continue
            
            try:
                r, p = pearsonr(eye_values[common_idx], img_values[common_idx])
                corr_results.at[stat, img_metric] = r
            except Exception:
                continue
    
    return corr_results


def compute_overall_correlation(df, eye_metric, stat_func='mean'):
    """Compute Pearson r pooling all categories together (no per-category split)."""
    agg_eye = getattr(df.groupby('image_name')[eye_metric], stat_func)()
    img_metrics_df = df.groupby('image_name')[IMAGE_METRICS].first()
    common_idx = agg_eye.index.intersection(img_metrics_df.index)
    agg_eye = agg_eye[common_idx]
    result = pd.Series(index=IMAGE_METRICS, dtype=float, name='pooled_r')
    for metric in IMAGE_METRICS:
        if metric not in img_metrics_df.columns:
            continue
        img_vals = img_metrics_df.loc[common_idx, metric]
        valid = agg_eye.notna() & img_vals.notna()
        if valid.sum() < 3:
            continue
        try:
            r, _ = pearsonr(agg_eye[valid], img_vals[valid])
            result[metric] = r
        except Exception:
            continue
    return result


def main():
    print("=" * 60)
    print("Target Verification Time Analysis")
    print("Prompt: plausible_realistic")
    print("=" * 60)
    
    print("\nLoading fixation data...")
    eye_data = load_fixation_data()
    print(f"Loaded {sum(len(imgs) for imgs in eye_data.values())} images across {len(eye_data)} categories")

    for blur_type in BLUR_TYPES:
        print(f"\n{'='*60}")
        print(f"Blur type: {blur_type}")
        print(f"{'='*60}")

        output_dir = BASE_OUTPUT_DIR / blur_type
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading image metrics from {PER_IMAGE_JSON_ROOT}...")
        image_metrics = load_image_metrics(blur_type)
        print(f"Loaded metrics for {sum(len(imgs) for imgs in image_metrics.values())} images")

        print("\nMatching eye tracking data with image metrics...")
        matched_df = match_data(eye_data, image_metrics)
        print(f"Matched {len(matched_df)} trials across {matched_df['image_name'].nunique()} images")

        if len(matched_df) == 0:
            print("ERROR: No matched data found!")
            continue

        excel_file = output_dir / "target_verification_time_correlations_by_statistic.xlsx"

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for stat_name in ['mean', 'median', 'min', 'max', 'std']:
                print(f"\nComputing {stat_name} correlations...")
                stat_corr = compute_correlations_by_category_with_aggregation(matched_df, stat_name)
                stat_corr.loc['pooled_r'] = compute_overall_correlation(matched_df, 'verification_time', stat_name)
                stat_corr.to_excel(writer, sheet_name=stat_name)

            raw_tab = matched_df[['category', 'image_name', 'verification_time']].copy()
            raw_tab = raw_tab.rename(columns={'verification_time': 'target_verification_time'})
            raw_tab = raw_tab.sort_values(['category', 'image_name']).reset_index(drop=True)
            raw_tab.to_excel(writer, sheet_name='raw_values', index=False)

        print(f"\nSaved: {excel_file}")
        print(f"  Sheets: mean, median, min, max, std (18 categories x 17 metrics each), raw_values")


if __name__ == "__main__":
    main()
