import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")
output_dir = Path("/home/kshaltiel/code/CSE-495-Code/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# Load fixations data
print("Loading fixations data...")
with open(fixations_path, 'r') as f:
    fixations_data = json.load(f)

# Create a mapping: (imageid, category) -> first fixation index (nfix)
# nfix = index of first fixation that lands inside the target bbox
fixations_map = {}
for entry in fixations_data:
    img_name = entry['name']
    imageid = Path(img_name).stem  # Remove .jpg extension
    category = entry['task']
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
                    first_idx = i
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

# Load metrics data and collect RCNN and YOLO confidences
print("Loading metrics data from per_image_metrics...")
data_points = []

# Dictionary to collect all conditions for same image-category pair
image_category_data = {}

for metrics_file in metrics_dir.glob("*.json"):
    try:
        # Parse filename: category_imageid_condition.json
        filename = metrics_file.stem
        parts = filename.split('_')
        
        # Extract category and image ID
        # Format: category_imageid_condition
        # e.g., bottle_000000019544_segmentation_no_blur
        category = parts[0]
        image_id = parts[1]
        condition = '_'.join(parts[2:])
        
        # Only process bbox_blur condition
        if condition != 'bbox_blur':
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Get RCNN and YOLO confidences (lists of 10 values)
        rcnn_confidences = metrics.get('rcnn_confidence', [])
        yolo_confidences = metrics.get('yolo_confidence', [])
        
        # Get nfix for this image-category pair (using imageid without .jpg)
        key = (image_id, category)
        nfix = fixations_avg.get(key, None)
        
        # Skip if no nfix data or if nfix is NaN
        if nfix is None or (isinstance(nfix, float) and np.isnan(nfix)):
            continue
        
        # Average the confidences across the 10 generations
        if rcnn_confidences:
            avg_rcnn = np.mean([c for c in rcnn_confidences if c is not None])
        else:
            avg_rcnn = 0
            
        if yolo_confidences:
            avg_yolo = np.mean([c for c in yolo_confidences if c is not None])
        else:
            avg_yolo = 0
        
        data_points.append({
            'category': category,
            'imageid': image_id,
            'condition': condition,
            'nfix': nfix,
            'rcnn_confidence': avg_rcnn,
            'yolo_confidence': avg_yolo
        })
        
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        continue

print(f"Collected {len(data_points)} data points")

# Convert to arrays for plotting
categories = [d['category'] for d in data_points]
nfix_vals = np.array([d['nfix'] for d in data_points])
rcnn_conf = np.array([d['rcnn_confidence'] for d in data_points])
yolo_conf = np.array([d['yolo_confidence'] for d in data_points])

# ============= SCATTERPLOT 1: nfix vs RCNN =============
plt.figure(figsize=(10, 6))
plt.scatter(rcnn_conf, nfix_vals, alpha=0.6, s=30, color='#bdd7ee', edgecolors='black', linewidths=0.5)
plt.xlabel('RCNN Confidence', fontsize=12)
plt.ylabel('Number of Fixations to Target (nfix)', fontsize=12)
plt.title('Number of Fixations to Target vs RCNN Confidence', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add correlation
corr_rcnn, p_rcnn = pearsonr(rcnn_conf, nfix_vals)
plt.text(0.95, 0.95, f'Pearson r = {corr_rcnn:.3f}\np = {p_rcnn:.4f}',
         transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'nfix_vs_rcnn_scatter.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'nfix_vs_rcnn_scatter.png'}")
plt.close()

# ============= SCATTERPLOT 2: nfix vs YOLO =============
plt.figure(figsize=(10, 6))
plt.scatter(yolo_conf, nfix_vals, alpha=0.6, s=30, color='#bdd7ee', edgecolors='black', linewidths=0.5)
plt.xlabel('YOLO Confidence', fontsize=12)
plt.ylabel('Number of Fixations to Target (nfix)', fontsize=12)
plt.title('Number of Fixations to Target vs YOLO Confidence', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add correlation
corr_yolo, p_yolo = pearsonr(yolo_conf, nfix_vals)
plt.text(0.95, 0.95, f'Pearson r = {corr_yolo:.3f}\np = {p_yolo:.4f}',
         transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'nfix_vs_yolo_scatter.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'nfix_vs_yolo_scatter.png'}")
plt.close()

# ============= BAR PLOT 1: Correlation per Category (RCNN) =============
# Calculate correlation for each category
category_correlations_rcnn = {}
category_p_values_rcnn = {}

unique_categories = sorted(set(categories))
for cat in unique_categories:
    # Get data points for this category
    mask = np.array([c == cat for c in categories])
    cat_nfix = nfix_vals[mask]
    cat_rcnn = rcnn_conf[mask]
    
    if len(cat_nfix) > 2:  # Need at least 3 points for correlation
        corr, p_val = pearsonr(cat_rcnn, cat_nfix)
        category_correlations_rcnn[cat] = corr
        category_p_values_rcnn[cat] = p_val
    else:
        category_correlations_rcnn[cat] = 0
        category_p_values_rcnn[cat] = 1.0

# Create bar plot
plt.figure(figsize=(12, 6))
# Sort categories by correlation value (ascending order)
sorted_cats = sorted(category_correlations_rcnn.items(), key=lambda x: x[1])
cats = [c[0] for c in sorted_cats]
corrs = [c[1] for c in sorted_cats]

bars = plt.bar(range(len(cats)), corrs, color='#bdd7ee', alpha=0.8, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
plt.ylabel('Pearson Correlation (r)', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.title('Correlation between RCNN Confidence and Number of Fixations to Target by Category', 
          fontsize=14, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(-0.3, 0.1)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_rcnn_by_category.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'correlation_rcnn_by_category.png'}")
plt.close()

# ============= BAR PLOT 2: Correlation per Category (YOLO) =============
# Calculate correlation for each category
category_correlations_yolo = {}
category_p_values_yolo = {}

for cat in unique_categories:
    # Get data points for this category
    mask = np.array([c == cat for c in categories])
    cat_nfix = nfix_vals[mask]
    cat_yolo = yolo_conf[mask]
    
    if len(cat_nfix) > 2:  # Need at least 3 points for correlation
        corr, p_val = pearsonr(cat_yolo, cat_nfix)
        category_correlations_yolo[cat] = corr
        category_p_values_yolo[cat] = p_val
    else:
        category_correlations_yolo[cat] = 0
        category_p_values_yolo[cat] = 1.0

# Create bar plot
plt.figure(figsize=(12, 6))
# Sort categories by correlation value (ascending order)
sorted_cats = sorted(category_correlations_yolo.items(), key=lambda x: x[1])
cats = [c[0] for c in sorted_cats]
corrs = [c[1] for c in sorted_cats]

bars = plt.bar(range(len(cats)), corrs, color='#bdd7ee', alpha=0.8, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
plt.ylabel('Pearson Correlation (r)', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.title('Correlation between YOLO Confidence and Number of Fixations to Target by Category', 
          fontsize=14, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(-0.3, 0.1)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_yolo_by_category.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'correlation_yolo_by_category.png'}")
plt.close()

# ============= PRINT SUMMARY STATISTICS =============
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal data points: {len(data_points)}")
print(f"Number of categories: {len(unique_categories)}")
print(f"Categories: {', '.join(unique_categories)}")

print(f"\n--- Overall Correlations ---")
print(f"RCNN vs nfix: r = {corr_rcnn:.4f}, p = {p_rcnn:.4f}")
print(f"YOLO vs nfix: r = {corr_yolo:.4f}, p = {p_yolo:.4f}")

print(f"\n--- Per-Category Correlations (RCNN) ---")
for cat in sorted(category_correlations_rcnn.keys()):
    mask = np.array([c == cat for c in categories])
    n_points = np.sum(mask)
    corr = category_correlations_rcnn[cat]
    p_val = category_p_values_rcnn[cat]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{cat:20s}: r = {corr:6.3f}, p = {p_val:.4f} {sig:3s} (n={n_points})")

print(f"\n--- Per-Category Correlations (YOLO) ---")
for cat in sorted(category_correlations_yolo.keys()):
    mask = np.array([c == cat for c in categories])
    n_points = np.sum(mask)
    corr = category_correlations_yolo[cat]
    p_val = category_p_values_yolo[cat]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{cat:20s}: r = {corr:6.3f}, p = {p_val:.4f} {sig:3s} (n={n_points})")

print("\n" + "="*60)
print("All visualizations saved to:", output_dir)
print("="*60)
