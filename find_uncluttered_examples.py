#!/usr/bin/env python3
"""Find high nfix + low RCNN examples that aren't too cluttered.
We'll use the number of objects detected in the original image as a proxy for clutter.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")
coco_annotations_train = Path("/home/kshaltiel/cluster_test_embed/coco_annotations/instances_train2017.json")
coco_annotations_val = Path("/home/kshaltiel/cluster_test_embed/coco_annotations/instances_val2017.json")

# Load COCO annotations to count objects per image
print("Loading COCO annotations...")
with open(coco_annotations_train, 'r') as f:
    coco_train = json.load(f)
with open(coco_annotations_val, 'r') as f:
    coco_val = json.load(f)

# Count annotations per image
image_object_counts = {}
for ann in coco_train['annotations']:
    img_id = ann['image_id']
    image_object_counts[img_id] = image_object_counts.get(img_id, 0) + 1

for ann in coco_val['annotations']:
    img_id = ann['image_id']
    image_object_counts[img_id] = image_object_counts.get(img_id, 0) + 1

# Create mapping from filename to image_id
filename_to_id = {}
for img in coco_train['images']:
    filename_to_id[img['file_name']] = img['id']
for img in coco_val['images']:
    filename_to_id[img['file_name']] = img['id']

print(f"Loaded object counts for {len(image_object_counts)} images")

# Load fixations data and calculate nfix
print("\nLoading fixations data...")
with open(fixations_path, 'r') as f:
    fixations_data = json.load(f)

fixations_map = {}
for entry in fixations_data:
    img_name = entry['name']
    imageid = Path(img_name).stem
    category = entry['task']
    bbox = entry.get('bbox')
    xs = entry.get('X', [])
    ys = entry.get('Y', [])
    
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
        if key not in fixations_map:
            fixations_map[key] = []
        fixations_map[key].append(np.nan)

# Average nfix per image
fixations_avg = {}
for k, lst in fixations_map.items():
    arr = np.array([x for x in lst if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if arr.size == 0:
        fixations_avg[k] = np.nan
    else:
        fixations_avg[k] = float(np.nanmean(arr))

print(f"Loaded {len(fixations_avg)} unique image-category pairs with nfix data")

# Load metrics for bbox_blur condition
print("\nLoading metrics data from per_image_metrics...")
data_points = []

for metrics_file in metrics_dir.glob("*.json"):
    try:
        filename = metrics_file.stem
        parts = filename.split('_')
        
        category = parts[0]
        image_id = parts[1]
        condition = '_'.join(parts[2:])
        
        if condition != 'bbox_blur':
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        rcnn_confidences = metrics.get('rcnn_confidence', [])
        
        key = (image_id, category)
        nfix = fixations_avg.get(key, None)
        
        if nfix is None or (isinstance(nfix, float) and np.isnan(nfix)):
            continue
        
        if rcnn_confidences:
            avg_rcnn = np.mean([c for c in rcnn_confidences if c is not None])
        else:
            avg_rcnn = 0
        
        # Get object count for this image
        img_filename = f"{image_id}.jpg"
        img_id = filename_to_id.get(img_filename, None)
        object_count = image_object_counts.get(img_id, 0) if img_id else 0
        
        data_points.append({
            'category': category,
            'imageid': image_id,
            'nfix': nfix,
            'rcnn_confidence': avg_rcnn,
            'object_count': object_count
        })
        
    except Exception as e:
        continue

print(f"Collected {len(data_points)} data points\n")

# Find high nfix + low RCNN examples with low object counts
print("="*80)
print("HIGH NFIX + LOW RCNN (with object counts)")
print("="*80)

high_nfix_low_rcnn = [d for d in data_points if d['nfix'] > 5 and d['rcnn_confidence'] < 0.5]
high_nfix_low_rcnn = sorted(high_nfix_low_rcnn, key=lambda x: (x['object_count'], -x['nfix']))

print("\nSorted by object count (least cluttered first):")
print("-" * 80)
for i, d in enumerate(high_nfix_low_rcnn[:20], 1):
    print(f"{i:2d}. {d['imageid']}.jpg | {d['category']:15s} | nfix: {d['nfix']:4.1f} | "
          f"RCNN: {d['rcnn_confidence']:.3f} | Objects: {d['object_count']:3d}")

# Also show stats
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

object_counts = [d['object_count'] for d in high_nfix_low_rcnn]
print(f"\nObject count distribution for high nfix + low RCNN examples:")
print(f"  Min: {min(object_counts)}")
print(f"  Max: {max(object_counts)}")
print(f"  Mean: {np.mean(object_counts):.1f}")
print(f"  Median: {np.median(object_counts):.1f}")

# Show examples with different clutter levels
print("\n" + "="*80)
print("RECOMMENDATIONS BY CLUTTER LEVEL")
print("="*80)

print("\n1. LOW CLUTTER (< 10 objects):")
print("-" * 80)
low_clutter = [d for d in high_nfix_low_rcnn if d['object_count'] < 10][:5]
for i, d in enumerate(low_clutter, 1):
    print(f"{i}. {d['imageid']}.jpg | {d['category']:15s} | nfix: {d['nfix']:4.1f} | "
          f"RCNN: {d['rcnn_confidence']:.3f} | Objects: {d['object_count']:3d}")

print("\n2. MEDIUM CLUTTER (10-20 objects):")
print("-" * 80)
med_clutter = [d for d in high_nfix_low_rcnn if 10 <= d['object_count'] < 20][:5]
for i, d in enumerate(med_clutter, 1):
    print(f"{i}. {d['imageid']}.jpg | {d['category']:15s} | nfix: {d['nfix']:4.1f} | "
          f"RCNN: {d['rcnn_confidence']:.3f} | Objects: {d['object_count']:3d}")

print("\n3. HIGH CLUTTER (20+ objects):")
print("-" * 80)
high_clutter = [d for d in high_nfix_low_rcnn if d['object_count'] >= 20][:5]
for i, d in enumerate(high_clutter, 1):
    print(f"{i}. {d['imageid']}.jpg | {d['category']:15s} | nfix: {d['nfix']:4.1f} | "
          f"RCNN: {d['rcnn_confidence']:.3f} | Objects: {d['object_count']:3d}")

print("\n" + "="*80)
print("\nImages are located at:")
print("  /home/kshaltiel/code/CSE-495-Code/COCO_IMAGES/<category>/<imageid>.jpg")
print("="*80)
