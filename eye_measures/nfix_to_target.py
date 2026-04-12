"""
nfix_to_target.py

Calculate number of fixations to first landing on target and correlate with image quality metrics.
Uses precomputed metrics from plausible_realistic prompt experiments.
Eye metric: first_target_index + 1 (1-based count of fixations to first target landing)
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
    x0, y0, w, h = bbox
    return x0 <= x <= x0 + w and y0 <= y <= y0 + h


def load_fixation_data():
    """Load fixation data and extract number of fixations to first target landing."""
    with open(FIXATION_DATA, 'r') as f:
        data = json.load(f)

    eye_data = defaultdict(lambda: defaultdict(list))

    for trial in data:
        # Only target-present, correct trials
        if trial.get('condition') != 'present' or trial.get('correct') != 1:
            continue

        X = trial.get('X', [])
        Y = trial.get('Y', [])
        bbox = trial.get('bbox', [])

        if len(bbox) != 4 or len(X) == 0:
            continue

        # Find first fixation on target
        first_target_index = None
        for i in range(len(X)):
            if point_in_bbox(X[i], Y[i], bbox):
                first_target_index = i
                break

        if first_target_index is None:
            continue

        # 1-based: 1 means target was found on the very first fixation
        nfix = first_target_index + 1

        image_name = trial['name']
        task = trial['task']
        eye_data[task][image_name].append(nfix)

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
            for eye_val in eye_data[category][image_name]:
                row = {
                    'category': category,
                    'image_name': image_name,
                    'nfix_to_target': eye_val
                }
                row.update(image_metrics[category][image_name])
                matched.append(row)

    return pd.DataFrame(matched)


def compute_correlations_by_category_with_aggregation(df, stat_func='mean'):
    """Compute correlations per category, aggregating eye metric values per image first.

    Args:
        df: DataFrame with columns: category, image_name, nfix_to_target, and image metrics
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
            agg_eye = cat_df.groupby('image_name')['nfix_to_target'].mean()
        elif stat_func == 'median':
            agg_eye = cat_df.groupby('image_name')['nfix_to_target'].median()
        elif stat_func == 'min':
            agg_eye = cat_df.groupby('image_name')['nfix_to_target'].min()
        elif stat_func == 'max':
            agg_eye = cat_df.groupby('image_name')['nfix_to_target'].max()
        elif stat_func == 'std':
            agg_eye = cat_df.groupby('image_name')['nfix_to_target'].std()
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
    print("Number of Fixations to Target Analysis")
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

        excel_file = output_dir / "nfix_to_target_correlations_by_statistic.xlsx"

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for stat_name in ['mean', 'median', 'min', 'max', 'std']:
                print(f"\nComputing {stat_name} correlations...")
                stat_corr = compute_correlations_by_category_with_aggregation(matched_df, stat_name)
                stat_corr.loc['pooled_r'] = compute_overall_correlation(matched_df, 'nfix_to_target', stat_name)
                stat_corr.to_excel(writer, sheet_name=stat_name)

            raw_tab = matched_df[['category', 'image_name', 'nfix_to_target']].copy()
            raw_tab = raw_tab.rename(columns={'nfix_to_target': 'nfix_to_target'})
            raw_tab = raw_tab.sort_values(['category', 'image_name']).reset_index(drop=True)
            raw_tab.to_excel(writer, sheet_name='raw_values', index=False)

        print(f"\nSaved: {excel_file}")
        print(f"  Sheets: mean, median, min, max, std (18 categories x 17 metrics each), raw_values")


if __name__ == "__main__":
    main()
