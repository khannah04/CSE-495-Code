#!/usr/bin/env python3
"""Create PDF with low clutter + high nfix + low RCNN examples."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")
coco_images_dir = Path("/home/kshaltiel/code/CSE-495-Code/COCO_IMAGES")
output_dir = Path("/home/kshaltiel/code/CSE-495-Code/output")
pdf_path = Path("/home/kshaltiel/code/CSE-495-Code/low_clutter_high_nfix.pdf")
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
print("Loading metrics data from per_image_metrics...")
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
            max_rcnn_idx = int(np.argmax(rcnn_confidences))
            max_rcnn = max(rcnn_confidences)
        else:
            avg_rcnn = 0
            max_rcnn_idx = 0
            max_rcnn = 0
        
        # Get object count for this image
        img_filename = f"{image_id}.jpg"
        img_id = filename_to_id.get(img_filename, None)
        object_count = image_object_counts.get(img_id, 0) if img_id else 0
        
        data_points.append({
            'category': category,
            'imageid': image_id,
            'nfix': nfix,
            'rcnn_confidence': avg_rcnn,
            'max_rcnn_idx': max_rcnn_idx,
            'max_rcnn': max_rcnn,
            'object_count': object_count
        })
        
    except Exception as e:
        continue

print(f"Collected {len(data_points)} data points")

# Find high nfix + low RCNN examples with low object counts
high_nfix_low_rcnn = [d for d in data_points if d['nfix'] > 5 and d['rcnn_confidence'] < 0.5]
high_nfix_low_rcnn = sorted(high_nfix_low_rcnn, key=lambda x: (x['object_count'], -x['nfix']))

# Filter for low clutter (< 10 objects)
low_clutter_examples = [d for d in high_nfix_low_rcnn if d['object_count'] < 10]

print(f"\nFound {len(low_clutter_examples)} low clutter examples")

def load_images(data_entry):
    """Load original image and best inpainted image."""
    imageid = data_entry['imageid']
    category = data_entry['category']
    best_idx = data_entry['max_rcnn_idx']
    
    # Load original image
    orig_path = coco_images_dir / category / f"{imageid}.jpg"
    if not orig_path.exists():
        print(f"  Warning: Original image not found: {orig_path}")
        return None, None
    
    # Load inpainted image
    inpaint_path = output_dir / category / imageid / "bbox_blur" / f"{imageid}_rep{best_idx}.jpg"
    if not inpaint_path.exists():
        print(f"  Warning: Inpainted image not found: {inpaint_path}")
        return None, None
    
    orig_img = Image.open(orig_path).convert('RGB')
    inpaint_img = Image.open(inpaint_path).convert('RGB')
    
    return orig_img, inpaint_img

# Create PDF
print(f"\nCreating PDF: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    for idx, data in enumerate(low_clutter_examples, 1):
        orig_img, inpaint_img = load_images(data)
        
        if orig_img is None or inpaint_img is None:
            continue
        
        # Create figure
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(
            f"LOW CLUTTER + HIGH NFIX + LOW RCNN (Example {idx}/{len(low_clutter_examples)})\n"
            f"{data['imageid']}.jpg - {data['category']}\n"
            f"nfix: {data['nfix']:.1f} | RCNN confidence: {data['max_rcnn']:.3f} | "
            f"Total objects in scene: {data['object_count']}",
            fontsize=14, fontweight='bold', y=0.98
        )
        
        # Original image
        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(orig_img)
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Inpainted image
        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(inpaint_img)
        ax2.set_title(f"Best Inpainted (rep {data['max_rcnn_idx']})", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Added: {data['imageid']}.jpg ({data['category']}) - {data['object_count']} objects, nfix: {data['nfix']:.1f}")

print(f"\n{'='*80}")
print(f"PDF created successfully: {pdf_path}")
print(f"Total pages: {len(low_clutter_examples)}")
print(f"{'='*80}")
