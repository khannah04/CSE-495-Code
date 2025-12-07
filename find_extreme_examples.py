#!/usr/bin/env python3
"""Find extreme examples of images based on nfix and detector confidence."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")

# Load fixations data and calculate nfix
print("Loading fixations data...")
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
        yolo_confidences = metrics.get('yolo_confidence', [])
        
        key = (image_id, category)
        nfix = fixations_avg.get(key, None)
        
        if nfix is None or (isinstance(nfix, float) and np.isnan(nfix)):
            continue
        
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
            'nfix': nfix,
            'rcnn_confidence': avg_rcnn,
            'yolo_confidence': avg_yolo
        })
        
    except Exception as e:
        continue

print(f"Collected {len(data_points)} data points\n")

# Find extreme examples
print("="*80)
print("EXTREME EXAMPLES")
print("="*80)

# 1. Low nfix + High RCNN confidence (easy to find + high detection)
print("\n1. LOW NFIX + HIGH RCNN CONFIDENCE (easy to find, detected well)")
print("-" * 80)
low_nfix_high_rcnn = [d for d in data_points if d['nfix'] < 5 and d['rcnn_confidence'] > 0.7]
low_nfix_high_rcnn = sorted(low_nfix_high_rcnn, key=lambda x: (x['nfix'], -x['rcnn_confidence']))[:10]

for i, d in enumerate(low_nfix_high_rcnn, 1):
    print(f"{i}. {d['imageid']}.jpg | Category: {d['category']:15s} | nfix: {d['nfix']:.1f} | RCNN: {d['rcnn_confidence']:.3f} | YOLO: {d['yolo_confidence']:.3f}")

# 2. High nfix + Low RCNN confidence (hard to find + poorly detected)
print("\n2. HIGH NFIX + LOW RCNN CONFIDENCE (hard to find, detected poorly)")
print("-" * 80)
high_nfix_low_rcnn = [d for d in data_points if d['nfix'] > 5 and d['rcnn_confidence'] < 0.5]
high_nfix_low_rcnn = sorted(high_nfix_low_rcnn, key=lambda x: (-x['nfix'], x['rcnn_confidence']))[:10]

for i, d in enumerate(high_nfix_low_rcnn, 1):
    print(f"{i}. {d['imageid']}.jpg | Category: {d['category']:15s} | nfix: {d['nfix']:.1f} | RCNN: {d['rcnn_confidence']:.3f} | YOLO: {d['yolo_confidence']:.3f}")

# 3. Low nfix + High YOLO confidence (easy to find + high detection)
print("\n3. LOW NFIX + HIGH YOLO CONFIDENCE (easy to find, detected well)")
print("-" * 80)
low_nfix_high_yolo = [d for d in data_points if d['nfix'] < 5 and d['yolo_confidence'] > 0.7]
low_nfix_high_yolo = sorted(low_nfix_high_yolo, key=lambda x: (x['nfix'], -x['yolo_confidence']))[:10]

for i, d in enumerate(low_nfix_high_yolo, 1):
    print(f"{i}. {d['imageid']}.jpg | Category: {d['category']:15s} | nfix: {d['nfix']:.1f} | RCNN: {d['rcnn_confidence']:.3f} | YOLO: {d['yolo_confidence']:.3f}")

# 4. High nfix + Low YOLO confidence (hard to find + poorly detected)
print("\n4. HIGH NFIX + LOW YOLO CONFIDENCE (hard to find, detected poorly)")
print("-" * 80)
high_nfix_low_yolo = [d for d in data_points if d['nfix'] > 5 and d['yolo_confidence'] < 0.5]
high_nfix_low_yolo = sorted(high_nfix_low_yolo, key=lambda x: (-x['nfix'], x['yolo_confidence']))[:10]

for i, d in enumerate(high_nfix_low_yolo, 1):
    print(f"{i}. {d['imageid']}.jpg | Category: {d['category']:15s} | nfix: {d['nfix']:.1f} | RCNN: {d['rcnn_confidence']:.3f} | YOLO: {d['yolo_confidence']:.3f}")

print("\n" + "="*80)
print("\nTo view these images, look for them in:")
print("  COCO dataset: /home/kshaltiel/code/CSE-495-Code/COCO_IMAGES/<category>/<imageid>.jpg")
print("  Or search COCO dataset online with image ID")
print("="*80)
