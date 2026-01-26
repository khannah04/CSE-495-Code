"""
Correlate prompt experiment metrics with nfix (first fixation index).
Creates Excel files with correlation results for each prompt strategy.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict

# -------------------- Config --------------------
metrics_root = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_METRICS")
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
output_root = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_CORRELATIONS")
output_root.mkdir(parents=True, exist_ok=True)

# -------------------- Load Fixations Data --------------------
print("Loading fixations data...")
with open(fixations_path, 'r') as f:
    fixations_data = json.load(f)

# Create a mapping: (imageid, category) -> average first fixation index (nfix)
fixations_map = {}
for entry in fixations_data:
    img_name = entry['name']
    imageid = Path(img_name).stem  # Remove .jpg extension
    category = entry['task'].lower()
    bbox = entry.get('bbox')
    xs = entry.get('X', [])
    ys = entry.get('Y', [])
    
    # Find first fixation inside bbox
    first_idx = None
    if bbox and xs and ys:
        x0, y0, w, h = bbox
        x1 = x0 + w
        y1 = y0 + h
        for i, (xx, yy) in enumerate(zip(xs, ys)):
            try:
                if x0 <= xx <= x1 and y0 <= yy <= y1:
                    first_idx = i + 1  # 1-indexed
                    break
            except Exception:
                continue
    
    key = (imageid, category)
    if first_idx is not None:
        if key not in fixations_map:
            fixations_map[key] = []
        fixations_map[key].append(first_idx)
    else:
        # Record as NaN if no fixation found in bbox
        if key not in fixations_map:
            fixations_map[key] = []
        fixations_map[key].append(np.nan)

# Average nfix per image (across multiple trials/subjects)
fixations_avg = {}
for k, lst in fixations_map.items():
    arr = np.array([x for x in lst if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if arr.size == 0:
        fixations_avg[k] = np.nan
    else:
        fixations_avg[k] = float(np.nanmean(arr))

print(f"Loaded {len(fixations_avg)} unique image-category pairs with nfix data")

# -------------------- Process Each Prompt Strategy --------------------
for excel_file in metrics_root.glob("*_metrics.xlsx"):
    prompt_name = excel_file.stem.replace("_metrics", "")
    print(f"\n{'='*60}")
    print(f"Processing prompt strategy: {prompt_name}")
    print(f"{'='*60}")
    
    # Read both sheets
    bbox_df = pd.read_excel(excel_file, sheet_name='bbox')
    seg_df = pd.read_excel(excel_file, sheet_name='segmentation')
    
    # Process bbox and segmentation separately
    for condition_name, df in [('bbox', bbox_df), ('segmentation', seg_df)]:
        print(f"\n  Processing {condition_name} condition...")
        
        # Average metrics across reps for each image (matching existing code approach)
        metadata_cols = ['category', 'image', 'rep']
        metric_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Group by category and image, average all metric columns
        df_avg = df.groupby(['category', 'image'])[metric_cols].mean().reset_index()
        
        # Add nfix column
        nfix_values = []
        for idx, row in df_avg.iterrows():
            category = row['category'].lower()
            image = row['image']
            key = (image, category)
            nfix = fixations_avg.get(key, np.nan)
            nfix_values.append(nfix)
        
        df_avg['nfix'] = nfix_values
        
        # Remove rows without nfix data
        df_with_nfix = df_avg[~df_avg['nfix'].isna()].copy()
        
        print(f"    Total unique images: {len(df_avg)}")
        print(f"    Images with nfix: {len(df_with_nfix)}")
        
        if len(df_with_nfix) < 3:
            print(f"    ⚠️ Not enough data points for correlation (need at least 3)")
            continue
        
        print(f"    Correlating {len(metric_cols)} metrics with nfix")
        
        # Compute correlations
        correlations = []
        
        for metric in metric_cols:
            if metric not in df_with_nfix.columns:
                continue
            
            # Get valid pairs (both metric and nfix are non-NaN)
            valid_mask = ~df_with_nfix[metric].isna() & ~df_with_nfix['nfix'].isna()
            
            if valid_mask.sum() < 3:
                correlations.append({
                    'metric': metric,
                    'r': np.nan,
                    'p': np.nan,
                    'n': int(valid_mask.sum())
                })
                continue
            
            metric_vals = df_with_nfix.loc[valid_mask, metric].values
            nfix_vals = df_with_nfix.loc[valid_mask, 'nfix'].values
            
            try:
                r_val, p_val = pearsonr(metric_vals, nfix_vals)
                correlations.append({
                    'metric': metric,
                    'r': float(r_val),
                    'p': float(p_val),
                    'n': int(valid_mask.sum())
                })
            except Exception as e:
                print(f"    ⚠️ Error computing correlation for {metric}: {e}")
                correlations.append({
                    'metric': metric,
                    'r': np.nan,
                    'p': np.nan,
                    'n': int(valid_mask.sum())
                })
        
        # Create correlation dataframe
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('r', ascending=False, key=lambda x: abs(x))
        
        # Also compute per-category correlations
        per_category_corrs = defaultdict(list)
        
        for category in df_with_nfix['category'].unique():
            cat_df = df_with_nfix[df_with_nfix['category'] == category]
            
            if len(cat_df) < 3:
                continue
            
            for metric in metric_cols:
                if metric not in cat_df.columns:
                    continue
                
                valid_mask = ~cat_df[metric].isna() & ~cat_df['nfix'].isna()
                
                if valid_mask.sum() < 3:
                    continue
                
                metric_vals = cat_df.loc[valid_mask, metric].values
                nfix_vals = cat_df.loc[valid_mask, 'nfix'].values
                
                try:
                    r_val, p_val = pearsonr(metric_vals, nfix_vals)
                    per_category_corrs[metric].append({
                        'category': category,
                        'r': float(r_val),
                        'p': float(p_val),
                        'n': int(valid_mask.sum())
                    })
                except Exception:
                    continue
        
        # Create per-category correlation dataframes
        per_cat_dfs = {}
        for metric, corr_list in per_category_corrs.items():
            if len(corr_list) > 0:
                per_cat_dfs[metric] = pd.DataFrame(corr_list).sort_values('r', ascending=False, key=lambda x: abs(x))
        
        # Save to Excel with multiple sheets
        output_file = output_root / f"{prompt_name}_{condition_name}_nfix_correlations.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Overall correlations
            corr_df.to_excel(writer, sheet_name='Overall', index=False)
            
            # Per-category correlations (one sheet per metric)
            for metric, cat_df in per_cat_dfs.items():
                # Truncate sheet name if too long (Excel limit is 31 chars)
                sheet_name = metric[:31] if len(metric) > 31 else metric
                cat_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"    ✅ Saved correlations to: {output_file}")
        print(f"       - Overall sheet with {len(corr_df)} metrics")
        print(f"       - {len(per_cat_dfs)} per-metric category sheets")
        
        # # Print top 5 strongest correlations
        # print(f"\n    Top 5 strongest correlations (by |r|):")
        # for idx, row in corr_df.head(5).iterrows():
        #     print(f"      {row['metric']:20s}: r={row['r']:7.4f}, p={row['p']:.4e}, n={row['n']}")

print(f"\n{'='*60}")
print("ALL CORRELATIONS COMPLETE!")
print(f"Results saved to: {output_root}")
print(f"{'='*60}")
