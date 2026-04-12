import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
BASE_STYLE = {
    "figure.figsize": (14, 7),
    "figure.titlesize": 18,
    "axes.titlesize": 18,
    "font.size": 18,
    "font.family": "Lato",
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 21,
    "axes.titlelocation": "center",
    "axes.titlepad": 20,
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "lines.linewidth": 3,
    "lines.markersize": 6,
    "figure.subplot.left":   0.1,
    "figure.subplot.right":  0.70,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top":    0.9,
}
plt.rcParams.update(BASE_STYLE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir    = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/plausible_realistic")
xlsx_dir       = Path("/home/kshaltiel/code/CSE-495-Code/metrics_calculations/outputs/bbox")
output_dir     = Path("/home/kshaltiel/code/CSE-495-Code/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# Set True to run only the combined-eye-metrics figures in Part 3.
RUN_ONLY_COMBINED = False 

# ---------------------------------------------------------------------------
# Key predictors to visualize and their display labels
# ---------------------------------------------------------------------------
KEY_PREDICTORS = [
    ("rcnn_confidence",  "Target Reconstructability (TR)"),
    ("yolo_confidence",  "YOLO Confidence"),
    ("high_L2",          "High-Level L2 Distance"),
    ("mid_L2",           "Mid-Level L2 Distance"),
    ("low_L2",           "Low-Level L2 Distance"),
    ("mid_cosine",       "Mid-Level Cosine Similarity"),
    ("high_cosine",      "High-Level Cosine Similarity"),
    ("clip_cosine",      "CLIP Cosine Similarity"),
    ("dino_cosine",      "DINO Cosine Similarity"),
    ("clip_text_similarity", "CLIP Text Similarity"),
]

EYE_METRIC_LABELS = {
    "nfix_to_target":              "Number of Fixations to Target",
    "first_saccade_initiation":    "First Saccade Initiation Time",
    "second_fix_to_target_landing":"Second Fixation to Target Landing",
    "target_verification_time":    "Target Verification Time",
    "total_search_time":           "Total Search Time",
}

# ---------------------------------------------------------------------------
# Helper: single bar plot
# ---------------------------------------------------------------------------
def make_bar_plot(cats, corrs, xlabel, ylabel, save_path, figsize=(28, 12)):
    y_vals = np.array(corrs, dtype=float)
    y_min = float(np.nanmin(y_vals))
    y_max = float(np.nanmax(y_vals))
    margin = 0.05
    lo = round(np.floor((y_min - margin) * 10) / 10, 1)
    hi = round(np.ceil((y_max  + margin) * 10) / 10, 1)
    ticks = np.arange(lo, hi + 1e-9, 0.1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(cats)), y_vals, color='#bdd7ee', alpha=0.8,
           edgecolor='black', linewidth=2.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=50)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=50)
    ax.set_ylim(lo, hi)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=50)
    ax.set_ylabel(ylabel, fontsize=50)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.45, top=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# PART 1 — Per-metric bar plots from pre-computed xlsx files
#           Each metric gets its own subfolder under visualizations/
# ---------------------------------------------------------------------------
print("\n=== PART 1: Per-metric bar plots from xlsx ===\n")

if not RUN_ONLY_COMBINED:
    for xlsx_file in sorted(xlsx_dir.glob("*_correlations_by_statistic.xlsx")):
        metric_name = xlsx_file.stem.replace("_correlations_by_statistic", "")
        metric_label = EYE_METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())

        metric_output_dir = output_dir / metric_name
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- {metric_name} ---")

        df = pd.read_excel(xlsx_file, index_col=0)

        # Individual bar plot per predictor
        for col_key, col_label in KEY_PREDICTORS:
            if col_key not in df.columns:
                continue

            corr_series = df[col_key].dropna()
            sorted_items = sorted(corr_series.items(), key=lambda x: x[1])
            cats  = [c[0] for c in sorted_items]
            corrs = [c[1] for c in sorted_items]

            make_bar_plot(
                cats=cats,
                corrs=corrs,
                xlabel="Category",
                ylabel=f"Pearson r\n({metric_label})",
                save_path=metric_output_dir / f"correlation_{col_key}_by_category.png",
            )

        # Combined figure: 3-column grid of all available predictors
        available = [(k, lbl) for k, lbl in KEY_PREDICTORS if k in df.columns]
        if available:
            ncols = 3
            nrows = (len(available) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(24, 8 * nrows))
            axes_flat = np.array(axes).flatten()

            for idx, (col_key, col_label) in enumerate(available):
                corr_series = df[col_key].dropna()
                sorted_items = sorted(corr_series.items(), key=lambda x: x[1])
                cats  = [c[0] for c in sorted_items]
                corrs = [c[1] for c in sorted_items]

                ax = axes_flat[idx]
                ax.bar(range(len(cats)), corrs, color='#bdd7ee', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=22)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel("Category", fontsize=22)
                ax.set_ylabel("Pearson r", fontsize=22)
                ax.tick_params(axis='y', labelsize=22)
                ax.set_title(col_label, fontsize=24)

            for j in range(len(available), len(axes_flat)):
                axes_flat[j].set_visible(False)

            plt.suptitle(f"{metric_label}\nCorrelations by Category", fontsize=26, y=1.01)
            plt.tight_layout()
            combined_path = metric_output_dir / "correlation_all_predictors_combined.png"
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {combined_path}")
            plt.close()

print("\n=== PART 1 complete ===\n")

# ---------------------------------------------------------------------------
# PART 2 — All eye metrics from raw JSON data
#           Each metric gets its own subfolder under visualizations/
# ---------------------------------------------------------------------------
print("=== PART 2: All eye metric scatter + bar plots from raw data ===\n")

# ---- Load fixation data and compute all 5 eye metrics at once ----
print("Loading fixations data and computing eye metrics...")
with open(fixations_path, 'r') as f:
    fixations_data = json.load(f)

from collections import defaultdict

# Storage: metric_name -> {(imageid, category): [values]}
raw_eye = {
    'nfix_to_target':               defaultdict(list),
    'first_saccade_initiation':     defaultdict(list),
    'second_fix_to_target_landing': defaultdict(list),
    'target_verification_time':     defaultdict(list),
    'total_search_time':            defaultdict(list),
}

def _in_bbox(xx, yy, x0, y0, w, h):
    return x0 <= xx <= x0 + w and y0 <= yy <= y0 + h

for trial in fixations_data:
    imageid  = Path(trial['name']).stem
    category = trial['task']
    T        = trial.get('T', [])
    X        = trial.get('X', [])
    Y        = trial.get('Y', [])
    bbox     = trial.get('bbox', [])
    RT       = trial.get('RT', None)
    correct  = trial.get('correct', 0)
    condition= trial.get('condition', '')
    key      = (imageid, category)

    has_bbox = len(bbox) == 4
    if has_bbox:
        x0, y0, w, h = bbox

    # Find first fixation on target (used by multiple metrics)
    first_target_idx = None
    if has_bbox and len(X) > 0:
        for i in range(len(X)):
            try:
                if _in_bbox(X[i], Y[i], x0, y0, w, h):
                    first_target_idx = i
                    break
            except Exception:
                pass

    # All metrics: TP (target present) condition only
    if condition != 'present':
        continue
    
    # nfix_to_target — correct trials where target was found
    if correct == 1 and first_target_idx is not None:
        raw_eye['nfix_to_target'][key].append(first_target_idx + 1)

    # first_saccade_initiation — T[0] is duration of first fixation
    if correct == 1 and len(T) > 0:
        raw_eye['first_saccade_initiation'][key].append(T[0])

    # second_fix_to_target_landing — sum of fixation durations from 2nd fix to landing on target
    # This is sum(T[1:first_target_idx]), time from second fixation onset to target landing onset
    if correct == 1 and first_target_idx is not None and first_target_idx >= 1:
        raw_eye['second_fix_to_target_landing'][key].append(sum(T[1:first_target_idx]))

    # target_verification_time — time from landing on target to response
    if correct == 1 and RT is not None and first_target_idx is not None:
        raw_eye['target_verification_time'][key].append(RT - sum(T[0:first_target_idx]))

    # total_search_time — sum of first saccade, target-landing, and verification components
    if correct == 1 and len(T) > 0 and RT is not None and first_target_idx is not None:
        first_saccade_time = T[0]
        second_fix_time = sum(T[1:first_target_idx]) if first_target_idx >= 1 else 0
        verification_time = RT - sum(T[0:first_target_idx])
        raw_eye['total_search_time'][key].append(
            first_saccade_time + second_fix_time + verification_time
        )

# Average across trials per image
eye_avgs = {}
for metric_name, data in raw_eye.items():
    eye_avgs[metric_name] = {}
    for key, vals in data.items():
        clean = [v for v in vals if v is not None and not np.isnan(float(v))]
        eye_avgs[metric_name][key] = float(np.mean(clean)) if clean else np.nan

print(f"Eye metric totals: " + ", ".join(
    f"{m}: {sum(1 for v in eye_avgs[m].values() if not np.isnan(v))}"
    for m in eye_avgs))

# ---- Load per-image predictor values from JSON ----
print("Loading per-image predictor metrics from JSON files...")

def safe_mean(lst):
    vals = [v for v in (lst or []) if v is not None]
    return float(np.mean(vals)) if vals else np.nan

# image_preds[(imageid, category)] = {pred_key: value}
image_preds = {}
for metrics_file in metrics_dir.glob("*_bbox.json"):
    try:
        parts    = metrics_file.stem.split('_')
        category = parts[0]
        image_id = parts[1]
        with open(metrics_file, 'r') as f:
            m = json.load(f)
        image_preds[(image_id, category)] = {
            'rcnn_confidence': safe_mean(m.get('rcnn_confidence', [])),
            'yolo_confidence': safe_mean(m.get('yolo_confidence', [])),
            'high_l2':         safe_mean(m.get('high_L2',   [])),
            'mid_l2':          safe_mean(m.get('mid_L2',    [])),
            'low_l2':          safe_mean(m.get('low_L2',    [])),
            'mid_cosine':      safe_mean(m.get('mid_cosine',[])),
            'high_cosine':     safe_mean(m.get('high_cosine',[])),
            'clip_cosine':     safe_mean(m.get('clip_cosine',[])),
            'dino_cosine':     safe_mean(m.get('dino_cosine',[])),
            'clip_text_similarity': safe_mean(m.get('clip_text_similarity', [])),
        }
    except Exception as e:
        print(f"  Error: {metrics_file.name}: {e}")

print(f"Loaded predictor data for {len(image_preds)} images")

# Predictors: (key_in_image_preds, display_label)
raw_predictors = [
    ('rcnn_confidence', 'Target Reconstructability (TR)'),
    ('yolo_confidence', 'YOLO Confidence'),
    ('high_l2',         'High-Level L2 Distance'),
    ('mid_l2',          'Mid-Level L2 Distance'),
    ('low_l2',          'Low-Level L2 Distance'),
    ('mid_cosine',      'Mid-Level Cosine Similarity'),
    ('high_cosine',     'High-Level Cosine Similarity'),
    ('clip_cosine',     'CLIP Cosine Similarity'),
    ('dino_cosine',     'DINO Cosine Similarity'),
    ('clip_text_similarity', 'CLIP Text Similarity'),
]

# ---- Helper: scatter with regression ----
def scatter_with_regression(x, y, xlabel, ylabel, save_path):
    plt.figure()
    plt.scatter(x, y, alpha=0.6, s=30, color='#bdd7ee',
                edgecolors='black', linewidths=0.5)
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, np.poly1d(z)(x_line), color='goldenrod', alpha=0.8)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    r, p = pearsonr(x, y)
    plt.text(0.95, 0.95, f'Pearson r = {r:.3f}\np = {p:.4f}',
             transform=ax.transAxes, va='top', ha='right', fontsize=22,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def scatter_with_regression_ax(ax, x, y, xlabel, ylabel, title):
    """Scatter + linear fit on a provided axis with Pearson r annotation."""
    ax.scatter(x, y, alpha=0.6, s=30, color='#bdd7ee',
           edgecolors='black', linewidths=0.5)
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color='goldenrod', alpha=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    r, p = pearsonr(x, y)
    ax.text(0.95, 0.95, f'Pearson r = {r:.3f}\np = {p:.4f}',
        transform=ax.transAxes, va='top', ha='right', fontsize=18,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- Loop over every eye metric ----
if not RUN_ONLY_COMBINED:
    for metric_name, metric_label in EYE_METRIC_LABELS.items():
        metric_output_dir = output_dir / metric_name
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {metric_name} ---")

        eye_map = eye_avgs[metric_name]

        # Build matched arrays
        rows = []
        for key, eye_val in eye_map.items():
            if np.isnan(eye_val):
                continue
            preds = image_preds.get(key)
            if preds is None:
                continue
            row = {'category': key[1], 'eye_val': eye_val}
            row.update(preds)
            rows.append(row)

        if len(rows) < 10:
            print(f"  Skipping (only {len(rows)} matched points)")
            continue

        print(f"  {len(rows)} matched data points")
        categories_m    = [r['category'] for r in rows]
        eye_vals        = np.array([r['eye_val'] for r in rows])
        unique_cats     = sorted(set(categories_m))

        # Build predictor arrays (skip predictors with too many NaNs)
        active_predictors = []
        pred_arrays = {}
        for pred_key, pred_label in raw_predictors:
            arr = np.array([r[pred_key] for r in rows], dtype=float)
            valid = ~np.isnan(arr)
            if valid.sum() < 10:
                continue
            pred_arrays[pred_key] = arr
            active_predictors.append((pred_key, pred_label))

        # --- Individual scatter plots ---
        for pred_key, pred_label in active_predictors:
            arr = pred_arrays[pred_key]
            valid = ~np.isnan(arr)
            scatter_with_regression(
                arr[valid], eye_vals[valid],
                xlabel=pred_label,
                ylabel=metric_label,
                save_path=metric_output_dir / f'{metric_name}_vs_{pred_key}_scatter.png',
            )

        # --- Combined scatter (L2 types, 1x3) ---
        l2_keys = [k for k in ['high_l2', 'mid_l2', 'low_l2'] if k in pred_arrays]
        l2_labels = {'high_l2': 'High-Level L2 Distance',
                     'mid_l2':  'Mid-Level L2 Distance',
                     'low_l2':  'Low-Level L2 Distance'}
        if len(l2_keys) == 3:
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            for ax, k in zip(axes, l2_keys):
                arr = pred_arrays[k]
                valid = ~np.isnan(arr)
                ax.scatter(arr[valid], eye_vals[valid], alpha=0.6, s=30,
                           color='#bdd7ee', edgecolors='black', linewidths=0.5)
                z = np.polyfit(arr[valid], eye_vals[valid], 1)
                x_line = np.linspace(arr[valid].min(), arr[valid].max(), 100)
                ax.plot(x_line, np.poly1d(z)(x_line), color='goldenrod', alpha=0.8)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel(l2_labels[k], labelpad=10)
                ax.set_ylabel(metric_label)
                r, p = pearsonr(arr[valid], eye_vals[valid])
                ax.text(0.95, 0.95, f'Pearson r = {r:.3f}\np = {p:.4f}',
                        transform=ax.transAxes, va='top', ha='right', fontsize=22,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.savefig(metric_output_dir / f'{metric_name}_vs_l2_combined.png',
                        dpi=300, bbox_inches='tight')
            print(f"  Saved: {metric_output_dir / f'{metric_name}_vs_l2_combined.png'}")
            plt.close()

        # --- Per-category correlation bar plots ---
        cat_corrs = {}
        for pred_key, _ in active_predictors:
            arr = pred_arrays[pred_key]
            cat_corrs[pred_key] = {}
            for cat in unique_cats:
                mask = np.array([c == cat for c in categories_m])
                valid = mask & ~np.isnan(arr)
                if valid.sum() > 2:
                    r, _ = pearsonr(arr[valid], eye_vals[valid])
                    cat_corrs[pred_key][cat] = r
                else:
                    cat_corrs[pred_key][cat] = 0.0

        # Individual bar plots
        for pred_key, pred_label in active_predictors:
            sorted_items = sorted(cat_corrs[pred_key].items(), key=lambda x: x[1])
            make_bar_plot(
                cats=[c[0] for c in sorted_items],
                corrs=[c[1] for c in sorted_items],
                xlabel='Category',
                ylabel=f'Pearson r\n({metric_label})',
                save_path=metric_output_dir / f'correlation_{pred_key}_by_category.png',
            )

        # Combined L2 bar plot (1x3)
        if len(l2_keys) == 3:
            fig, axes = plt.subplots(1, 3, figsize=(42, 12))
            for ax, k in zip(axes, l2_keys):
                sorted_items = sorted(cat_corrs[k].items(), key=lambda x: x[1])
                cats_s  = [c[0] for c in sorted_items]
                corrs_s = [c[1] for c in sorted_items]
                ax.bar(range(len(cats_s)), corrs_s, color='#bdd7ee', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
                ax.set_xticks(range(len(cats_s)))
                ax.set_xticklabels(cats_s, rotation=45, ha='right', fontsize=22)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Category', fontsize=22, labelpad=10)
                ax.set_ylabel('Pearson r', fontsize=22)
                ax.tick_params(axis='y', labelsize=22)
                ax.set_title(l2_labels[k], fontsize=24)
            plt.subplots_adjust(bottom=0.35, wspace=0.3)
            plt.savefig(metric_output_dir / 'correlation_l2_by_category_combined.png',
                        dpi=300, bbox_inches='tight')
            print(f"  Saved: {metric_output_dir / 'correlation_l2_by_category_combined.png'}")
            plt.close()

        # Combined all-predictors bar plot (2x3 or 3x3)
        ncols = 3
        nrows = (len(active_predictors) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(42, 12 * nrows))
        axes_flat = np.array(axes).flatten()
        for idx, (pred_key, pred_label) in enumerate(active_predictors):
            sorted_items = sorted(cat_corrs[pred_key].items(), key=lambda x: x[1])
            cats_s  = [c[0] for c in sorted_items]
            corrs_s = [c[1] for c in sorted_items]
            ax = axes_flat[idx]
            ax.bar(range(len(cats_s)), corrs_s, color='#bdd7ee', alpha=0.8,
                   edgecolor='black', linewidth=2.5)
            ax.set_xticks(range(len(cats_s)))
            ax.set_xticklabels(cats_s, rotation=45, ha='right', fontsize=22)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Category', fontsize=22, labelpad=10)
            ax.set_ylabel('Pearson r', fontsize=22)
            ax.tick_params(axis='y', labelsize=22)
            ax.set_title(pred_label, fontsize=26, pad=12)
        for j in range(len(active_predictors), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        plt.savefig(metric_output_dir / 'correlation_all_predictors_combined.png',
                    dpi=300, bbox_inches='tight')
        print(f"  Saved: {metric_output_dir / 'correlation_all_predictors_combined.png'}")
        plt.close()

print("\n=== PART 2 complete ===")


# ---------------------------------------------------------------------------
# PART 3 — Combined across eye metrics: correlation spreadsheets
#          1) all 5 eye metrics vs RCNN confidence
#          2) all 5 eye metrics vs CLIP text similarity
# ---------------------------------------------------------------------------
print("\n=== PART 3: Combined across eye metrics (RCNN + CLIP text sim) ===\n")

combined_dir = output_dir / "combined_eye_metrics"
combined_dir.mkdir(parents=True, exist_ok=True)


def build_combined_eye_metric_from_spreadsheets(predictor_key, predictor_label, save_name):
    """Create a 2x3 panel figure for all eye metrics vs a predictor using spreadsheet correlations."""
    SPREADSHEET_EYE_METRICS = [
        "nfix_to_target",
        "first_saccade_initiation",
        "second_fix_to_target_landing",
        "target_verification_time",
        "total_search_time",
    ]
    
    ncols = 3
    nrows = (len(SPREADSHEET_EYE_METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(28, 10 * nrows))
    axes_flat = np.array(axes).flatten()

    plotted = 0
    for metric_idx, eye_metric in enumerate(SPREADSHEET_EYE_METRICS):
        xlsx_path = xlsx_dir / f"{eye_metric}_correlations_by_statistic.xlsx"
        if not xlsx_path.exists():
            axes_flat[metric_idx].set_visible(False)
            continue

        try:
            df_corr = pd.read_excel(xlsx_path, sheet_name="mean", index_col=0)
            df_corr = df_corr[~(df_corr.index.astype(str).str.lower() == "pooled_r")]
            if predictor_key not in df_corr.columns:
                axes_flat[metric_idx].set_visible(False)
                continue

            corr_series = df_corr[predictor_key].dropna()
            if len(corr_series) < 3:
                axes_flat[metric_idx].set_visible(False)
                continue

            ax = axes_flat[metric_idx]
            
            # Special handling for nfix_to_target: sort by average nfix, use bars with text labels
            if eye_metric == "nfix_to_target":
                # Compute average nfix per category
                nfix_by_cat = {}
                for (img_id, cat), val in eye_avgs['nfix_to_target'].items():
                    if not np.isnan(val):
                        if cat not in nfix_by_cat:
                            nfix_by_cat[cat] = []
                        nfix_by_cat[cat].append(val)
                
                avg_nfix_per_cat = {cat: np.mean(vals) for cat, vals in nfix_by_cat.items()}
                
                # Sort categories by average nfix (descending)
                sorted_cats = sorted(avg_nfix_per_cat.keys(), key=lambda c: avg_nfix_per_cat[c], reverse=True)
                
                # Get correlations for sorted categories
                corrs_sorted = [corr_series.get(cat, np.nan) for cat in sorted_cats]
                
                # Bar plot with text labels
                x = np.arange(len(sorted_cats))
                ax.bar(x, corrs_sorted, color='#bdd7ee', alpha=0.8, edgecolor='black', linewidth=2.5)
                
                # Add Pearson r text on top of each bar
                for i, (cat, r_val) in enumerate(zip(sorted_cats, corrs_sorted)):
                    if not np.isnan(r_val):
                        y_pos = r_val + 0.02 if r_val >= 0 else r_val - 0.08
                        ax.text(i, y_pos, f'{r_val:.2f}', ha='center', va='bottom' if r_val >= 0 else 'top',
                               fontsize=18, fontweight='bold')
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_cats, rotation=45, ha='right', fontsize=20)
                ax.set_xlabel("Category (sorted by avg nfix)", fontsize=20)
                ax.set_ylabel("Pearson r", fontsize=20)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(False)
                ax.tick_params(axis='y', labelsize=18)
            
            else:
                # Original scatter plot for other metrics
                sorted_items = sorted(corr_series.items(), key=lambda x: x[1])
                cats = [k for k, _ in sorted_items]
                vals = np.array([v for _, v in sorted_items], dtype=float)

                x = np.arange(len(cats))
                ax.scatter(x, vals, alpha=0.85, s=80, color="#bdd7ee", edgecolors="black", linewidths=1.2)
                ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
                ax.set_xticks(x)
                ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=20)
                ax.set_xlabel("Category", fontsize=20)
                ax.set_ylabel("Pearson r", fontsize=20)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.grid(False)
                ax.tick_params(axis="y", labelsize=18)
            
            ax.set_title(EYE_METRIC_LABELS.get(eye_metric, eye_metric), fontsize=22)
            plotted += 1

        except Exception as e:
            print(f"  Error reading {xlsx_path.name}: {e}")
            axes_flat[metric_idx].set_visible(False)
            continue

    for j in range(len(SPREADSHEET_EYE_METRICS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if plotted == 0:
        plt.close(fig)
        print(f"  Skipped: {save_name} (no valid panels)")
        return

    plt.suptitle(f"All Eye Metrics vs {predictor_label}", fontsize=26, y=1.00)
    plt.tight_layout()
    save_path = combined_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


build_combined_eye_metric_from_spreadsheets(
    predictor_key="rcnn_confidence",
    predictor_label="RCNN Confidence (Target Reconstructability)",
    save_name="all_eye_metrics_vs_rcnn_combined.png",
)

build_combined_eye_metric_from_spreadsheets(
    predictor_key="clip_text_similarity",
    predictor_label="CLIP Text Similarity",
    save_name="all_eye_metrics_vs_clip_text_similarity_combined.png",
)

print("\n=== PART 3 complete ===")

print(f"\n=== All done! Outputs in: {output_dir} ===")
print("Subfolders created:")
for d in sorted(output_dir.iterdir()):
    if d.is_dir():
        n = len(list(d.glob("*.png")))
        print(f"  {d.name}/  ({n} plots)")
