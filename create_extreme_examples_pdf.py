#!/usr/bin/env python3
"""Create PDF with extreme examples showing original and best inpainted images."""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths
fixations_path = Path("/home/kshaltiel/code/CSE-495-Code/coco_search18_fixations_TP_train_split1.json")
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")
coco_images_dir = Path("/home/kshaltiel/code/CSE-495-Code/COCO_IMAGES")
output_dir = Path("/home/kshaltiel/code/CSE-495-Code/output")
pdf_path = Path("/home/kshaltiel/code/CSE-495-Code/extreme_examples.pdf")

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
            max_rcnn_idx = int(np.argmax(rcnn_confidences))
            max_rcnn = max(rcnn_confidences)
        else:
            avg_rcnn = 0
            max_rcnn_idx = 0
            max_rcnn = 0
            
        if yolo_confidences:
            avg_yolo = np.mean([c for c in yolo_confidences if c is not None])
            max_yolo_idx = int(np.argmax(yolo_confidences))
            max_yolo = max(yolo_confidences)
        else:
            avg_yolo = 0
            max_yolo_idx = 0
            max_yolo = 0
        
        data_points.append({
            'category': category,
            'imageid': image_id,
            'nfix': nfix,
            'rcnn_confidence': avg_rcnn,
            'yolo_confidence': avg_yolo,
            'max_rcnn_idx': max_rcnn_idx,
            'max_yolo_idx': max_yolo_idx,
            'max_rcnn': max_rcnn,
            'max_yolo': max_yolo
        })
        
    except Exception as e:
        continue

print(f"Collected {len(data_points)} data points\n")

# Find extreme examples (top 5 each)
print("Finding extreme examples...")

# 1. Low nfix + High RCNN
low_nfix_high_rcnn = [d for d in data_points if d['nfix'] < 5 and d['rcnn_confidence'] > 0.7]
low_nfix_high_rcnn = sorted(low_nfix_high_rcnn, key=lambda x: (x['nfix'], -x['rcnn_confidence']))

# 2. High nfix + Low RCNN
high_nfix_low_rcnn = [d for d in data_points if 4 <= d['nfix'] <= 9 and d['rcnn_confidence'] < 0.5]
high_nfix_low_rcnn = sorted(high_nfix_low_rcnn, key=lambda x: (-x['nfix'], x['rcnn_confidence']))

# 3. Low nfix + High YOLO
low_nfix_high_yolo = [d for d in data_points if d['nfix'] < 5 and d['yolo_confidence'] > 0.7]
low_nfix_high_yolo = sorted(low_nfix_high_yolo, key=lambda x: (x['nfix'], -x['yolo_confidence']))

# 4. High nfix + Low YOLO
high_nfix_low_yolo = [d for d in data_points if 4 <= d['nfix'] <= 9 and d['yolo_confidence'] < 0.5]
high_nfix_low_yolo = sorted(high_nfix_low_yolo, key=lambda x: (-x['nfix'], x['yolo_confidence']))

# Create PDF
print(f"\nCreating PDF: {pdf_path}")

def load_images(data_entry, use_rcnn=True):
    """Load original image and best inpainted image."""
    imageid = data_entry['imageid']
    category = data_entry['category']
    best_idx = data_entry['max_rcnn_idx'] if use_rcnn else data_entry['max_yolo_idx']
    
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

def add_page(pdf, data_list, title, detector_name, use_rcnn=True):
    """Add a page to the PDF with examples."""
    if not data_list:
        return
    
    for data in data_list:
        orig_img, inpaint_img = load_images(data, use_rcnn=use_rcnn)
        
        if orig_img is None or inpaint_img is None:
            continue
        
        # Create figure
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        conf_val = data['max_rcnn'] if use_rcnn else data['max_yolo']
        fig.suptitle(
            f"{title}\n{data['imageid']}.jpg - {data['category']}\n"
            f"nfix: {data['nfix']:.1f} | {detector_name} confidence: {conf_val:.3f}",
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
        ax2.set_title(f"Best Inpainted (rep {data['max_rcnn_idx'] if use_rcnn else data['max_yolo_idx']})", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Added: {data['imageid']}.jpg ({data['category']})")

with PdfPages(pdf_path) as pdf:
    # Page 1: Low nfix + High RCNN
    print("\nAdding: Low nfix + High RCNN...")
    add_page(pdf, low_nfix_high_rcnn, 
             "LOW NFIX + HIGH RCNN\n(Easy to find, well detected)", 
             "RCNN", use_rcnn=True)
    
    # Page 2: High nfix + Low RCNN
    print("\nAdding: High nfix + Low RCNN...")
    add_page(pdf, high_nfix_low_rcnn, 
             "HIGH NFIX + LOW RCNN\n(Hard to find, poorly detected)", 
             "RCNN", use_rcnn=True)
    
    # Page 3: Low nfix + High YOLO
    print("\nAdding: Low nfix + High YOLO...")
    add_page(pdf, low_nfix_high_yolo, 
             "LOW NFIX + HIGH YOLO\n(Easy to find, well detected)", 
             "YOLO", use_rcnn=False)
    
    # Page 4: High nfix + Low YOLO
    print("\nAdding: High nfix + Low YOLO...")
    add_page(pdf, high_nfix_low_yolo, 
             "HIGH NFIX + LOW YOLO\n(Hard to find, poorly detected)", 
             "YOLO", use_rcnn=False)

print(f"\n{'='*80}")
print(f"PDF created successfully: {pdf_path}")
print(f"{'='*80}")
