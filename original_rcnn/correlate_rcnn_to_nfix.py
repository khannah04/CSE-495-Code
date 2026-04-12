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
rcnn_detections_path = Path("/home/kshaltiel/code/CSE-495-Code/rcnn_target_bbox_detections.json")
output_dir = Path("/home/kshaltiel/code/CSE-495-Code/rcnn_nfix_correlations")
output_dir.mkdir(exist_ok=True)

# Load fixations and compute average nfix per image
print("Loading fixation data...")
with open(fixations_path) as f:
    fixations = json.load(f)

nfix_by_image = {}
for trial in fixations:
    img_name = Path(trial['name']).stem + '.jpg'
    category = trial['task']
    key = (img_name, category)
    
    if trial.get('condition') != 'present' or trial.get('correct') != 1:
        continue
    
    # Find first fixation on target
    bbox = trial.get('bbox', [])
    X = trial.get('X', [])
    Y = trial.get('Y', [])
    
    if len(bbox) != 4 or len(X) == 0:
        continue
    
    x0, y0, w, h = bbox
    first_target_idx = None
    for i in range(len(X)):
        if x0 <= X[i] <= x0 + w and y0 <= Y[i] <= y0 + h:
            first_target_idx = i
            break
    
    if first_target_idx is not None:
        nfix_val = first_target_idx + 1
        if key not in nfix_by_image:
            nfix_by_image[key] = []
        nfix_by_image[key].append(nfix_val)

# Average nfix per image
avg_nfix = {k: np.mean(v) for k, v in nfix_by_image.items()}
print(f"Computed avg nfix for {len(avg_nfix)} image-category pairs")

# Load RCNN detections
print("Loading RCNN detections...")
with open(rcnn_detections_path) as f:
    rcnn_data = json.load(f)

# Extract scores per image - ONLY for correct category matches
rcnn_scores = {}
for entry_key, data in rcnn_data.items():
    # entey_key is in format "imagename.jpg_category"
    # Split from the right to get category
    parts = entry_key.rsplit('_', 1)
    if len(parts) != 2:
        continue
    
    img_name = parts[0]  # e.g., "000000478726.jpg"
    category = parts[1]  # e.g., "bottle"
    
    # Verify this matches the task field in data
    if data.get('task') != category:
        print(f"Warning: key mismatch for {entry_key}")
    
    key = (img_name, category)
    
    # Get scores from detections
    scores = data['detections']['scores']
    
    # Use the max score available (treating all detected objects equally)
    if scores and len(scores) > 0:
        rcnn_scores[key] = max(scores)
    else:
        # If no detections, assign 0
        rcnn_scores[key] = 0.0

print(f"Loaded RCNN scores for {len(rcnn_scores)} images")
print(f"Images with detections: {sum(1 for v in rcnn_scores.values() if v > 0)}")

# Match and compute correlations per category
print("Computing correlations per category...")
correlations = {}
category_data = defaultdict(lambda: {'scores': [], 'nfix': []})

for key, rcnn_score in rcnn_scores.items():
    if key in avg_nfix:
        img_name, category = key
        category_data[category]['scores'].append(rcnn_score)
        category_data[category]['nfix'].append(avg_nfix[key])

# Calculate Pearson correlation per category
results = {}
for category, data in sorted(category_data.items()):
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

for key, rcnn_score in rcnn_scores.items():
    if key in avg_nfix:
        all_scores.append(rcnn_score)
        all_nfix.append(avg_nfix[key])



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
