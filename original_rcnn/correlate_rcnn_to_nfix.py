"""
Correlate RCNN detection scores to average nfix per category.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
rcnn_detections_path = Path("/home/kshaltiel/code/CSE-495-Code/rcnn code/rcnn_target_bbox_detections.json")
output_dir = Path("/home/kshaltiel/code/CSE-495-Code/rcnn_nfix_correlations")
output_dir.mkdir(exist_ok=True)
def point_in_bbox(x, y, bbox):
    """Check if point (x, y) is inside bbox [x, y, width, height]."""
    x0, y0, w, h = bbox
    return x0 <= x <= x0 + w and y0 <= y <= y0 + h

# Load fixations and compute average nfix per image
print("Loading fixation data...")
with open(fixations_path) as f:
    fixations = json.load(f)

# Track which image-category pairs are present/correct for filtering RCNN scores
valid_keys = set()

nfix_by_image = {}
for trial in fixations:
    img_name = Path(trial['name']).stem + '.jpg'
    category = trial['task']
    key = (img_name, category)
    
    if trial.get('condition') != 'present' or trial.get('correct') != 1:
        continue
    
    # Mark as valid for RCNN filtering
    valid_keys.add(key)
    
    # Find first fixation on target
    bbox = trial.get('bbox', [])
    X = trial.get('X', [])
    Y = trial.get('Y', [])
    
    if len(bbox) != 4 or len(X) == 0:
        continue
    
    # Find first fixation on target
    first_target_idx = None
    for i in range(len(X)):
        if point_in_bbox(X[i], Y[i], bbox):
            first_target_idx = i
            break
    
    if first_target_idx is None:
        continue
    
    nfix_val = first_target_idx + 1
    if key not in nfix_by_image:
        nfix_by_image[key] = []
    nfix_by_image[key].append(nfix_val)

def load_rcnn_scores(valid_keys):
    """Load RCNN detection scores from JSON file, filtered to valid present/correct trials."""
    print("Loading RCNN detections...")
    with open(rcnn_detections_path) as f:
        rcnn_data = json.load(f)

    rcnn_scores = {}
    for entry_key, data in rcnn_data.items():
        # entry_key is in format "imagename.jpg_category"
        parts = entry_key.rsplit('_', 1)
        if len(parts) != 2:
            continue
        
        img_name = parts[0]
        category = parts[1]
        
        # Normalize and verify category matches task (case-insensitive)
        task_in_data = data.get('task', '').lower().strip()
        category_normalized = category.lower().strip()
        
        # Only validate if both are present and non-empty
        if task_in_data and category_normalized != task_in_data:
            continue
        
        key = (img_name, category)
        
        # Only include RCNN scores for valid present/correct trials
        if key not in valid_keys:
            continue
        
        # Safely extract target category score from nested dict
        detections = data.get('detections', {})
        if isinstance(detections, dict):
            target_score = detections.get('target_category_score', 0.0)
        else:
            target_score = 0.0
        
        rcnn_scores[key] = float(target_score)
    
    return rcnn_scores


def match_rcnn_and_nfix(rcnn_scores, nfix_by_image):
    """Match RCNN scores with nfix values."""
    matched_pairs = defaultdict(lambda: {'scores': [], 'nfix': []})
    
    for key, rcnn_score in rcnn_scores.items():
        if key in nfix_by_image:
            img_name, category = key
            # Average nfix per image
            avg_nfix_val = np.mean(nfix_by_image[key])
            matched_pairs[category]['scores'].append(rcnn_score)
            matched_pairs[category]['nfix'].append(avg_nfix_val)
    
    return matched_pairs


def compute_rcnn_correlations(matched_pairs):
    """Compute Pearson correlations per category."""
    results = {}
    for category in sorted(matched_pairs.keys()):
        data = matched_pairs[category]
        if len(data['scores']) > 2:
            r, p = pearsonr(data['scores'], data['nfix'])
            results[category] = {
                'r': float(r),
                'p': float(p),
                'n_samples': len(data['scores']),
                'mean_nfix': float(np.mean(data['nfix'])),
                'mean_score': float(np.mean(data['scores']))
            }
            print(f"  {category:20} r={r:6.3f} p={p:.4f} n={len(data['scores'])}")
        else:
            print(f"  {category:20} SKIPPED (n={len(data['scores'])})")
    
    return results


# Average nfix per image
avg_nfix = {k: np.mean(v) for k, v in nfix_by_image.items()}
print(f"Computed avg nfix for {len(avg_nfix)} image-category pairs")

# Load RCNN scores (filtered to valid present/correct trials)
rcnn_scores = load_rcnn_scores(valid_keys)
print(f"Loaded RCNN scores for {len(rcnn_scores)} images")
print(f"Images with detections: {sum(1 for v in rcnn_scores.values() if v > 0)}")

# Match and compute correlations per category
print("Computing correlations per category...")
matched_pairs = match_rcnn_and_nfix(rcnn_scores, nfix_by_image)
results = compute_rcnn_correlations(matched_pairs)

# Save correlations to JSON
output_json = output_dir / "rcnn_nfix_correlations.json"
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved correlations to {output_json}")

# Create bar plot
print("Creating bar plot...")
cats = list(results.keys())
r_vals = [results[c]['r'] for c in cats]

fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(range(len(cats)), r_vals, color='#bdd7ee', alpha=0.8, edgecolor='black', linewidth=2.5)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=40)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_ylabel('Pearson r', fontsize=40)
ax.set_xlabel('Category', fontsize=40)
ax.tick_params(axis='y', labelsize=35)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.25, top=0.95)
output_plot = output_dir / "rcnn_nfix_correlation_by_category.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_plot}")
plt.close()

# Create overall scatter plot
print("Creating overall scatter plot...")
all_scores = []
all_nfix = []

for category in matched_pairs.keys():
    data = matched_pairs[category]
    all_scores.extend(data['scores'])
    all_nfix.extend(data['nfix'])



all_scores = np.array(all_scores)
all_nfix = np.array(all_nfix)

print(f"Total matched pairs: {len(all_scores)}")

if len(all_scores) < 2:
    print("ERROR: Not enough matched data points to compute correlation!")
    print(f"n_scores={len(all_scores)}, n_nfix={len(all_nfix)}")
    exit(1)

print(f"All scores: {all_scores[:10]}...")
print(f"All nfix: {all_nfix[:10]}...")

# Compute overall correlation
r_overall, p_overall = pearsonr(all_scores, all_nfix)

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(all_scores, all_nfix, alpha=0.5, s=50, color='#bdd7ee', edgecolors='black', linewidths=0.5)

# Add regression line
z = np.polyfit(all_scores, all_nfix, 1)
x_line = np.linspace(all_scores.min(), all_scores.max(), 100)
ax.plot(x_line, np.poly1d(z)(x_line), color='goldenrod', alpha=0.8, linewidth=3)

ax.set_xlabel('RCNN Score', fontsize=40)
ax.set_ylabel('Average Nfix', fontsize=40)
ax.tick_params(axis='both', labelsize=35)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add stats
stats_text = f'Pearson r = {r_overall:.3f}\np = {p_overall:.4f}\nn = {len(all_scores)}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right', 
        fontsize=35, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.95)
output_scatter = output_dir / "rcnn_nfix_overall_scatter.png"
plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
print(f"Saved scatter plot to {output_scatter}")
print(f"Overall correlation: r={r_overall:.3f}, p={p_overall:.4f}")
plt.close()

print("\nDone!")
