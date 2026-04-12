import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import argparse
from tqdm import tqdm

# Configuration
ANNOTATIONS_FILE = 'coco_search18_fixations_TP_train_split1.json'
IMAGES_ROOT = '/home/kshaltiel/cluster_test_embed/coco_search_18_TP/images'
OUTPUT_FILE = 'rcnn_target_bbox_detections.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load COCO class names directly from model (FIXED)
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
COCO_CLASSES = weights.meta["categories"]

# COCO Search 18 classes (kept as reference filter)
COCO_SEARCH18_CLASSES = {
    3: 'car', 12: 'stop sign', 38: 'bottle', 40: 'cup', 41: 'fork', 
    42: 'knife', 44: 'bowl', 56: 'chair', 58: 'potted plant', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 66: 'keyboard', 67: 'microwave', 
    68: 'oven', 70: 'sink', 73: 'clock'
}

# Robust category normalization (FIXED)
CATEGORY_MAP = {
    "tv monitor": "tv",
    "plant": "potted plant",
    "sofa": "couch",
    "couch": "couch",
}

def normalize_category(name: str) -> str:
    name = name.lower().strip()
    return CATEGORY_MAP.get(name, name)

# Transform (FIXED: not recreated each call)
transform = transforms.ToTensor()

def load_model(device: torch.device):
    print("Loading Mask R-CNN model...")
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def load_annotations(annot_file: str) -> List[Dict[str, Any]]:
    print(f"Loading annotations from {annot_file}...")
    with open(annot_file, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")
    return annotations

def get_image_path(img_name: str, task_category: str, images_root: str) -> str:
    return os.path.join(images_root, task_category, img_name)

def crop_bbox(image: Image.Image, bbox: List[int]) -> Image.Image:
    """
    Crop image to bbox region.
    bbox: [x, y, width, height] format
    """
    x, y, width, height = bbox
    x = max(0, x)
    y = max(0, y)
    width = max(1, width)
    height = max(1, height)
    
    x2 = min(image.width, x + width)
    y2 = min(image.height, y + height)
    
    if x >= image.width or y >= image.height or x2 <= x or y2 <= y:
        return image
    
    return image.crop((x, y, x2, y2))

def run_rcnn_on_image(image: Image.Image, model, device: torch.device, target_category: str = None) -> Dict[str, Any]:
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    result = {
        'boxes': [],
        'scores': [],
        'labels': [],
        'label_names': [],
        'target_category': target_category,
        'target_category_score': 0.0,
    }

    if len(outputs) > 0:
        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        target_scores = []
        target_norm = normalize_category(target_category) if target_category else None

        for box, score, label_idx in zip(boxes, scores, labels):
            label_idx = int(label_idx)
            label_name = COCO_CLASSES[label_idx]
            label_norm = normalize_category(label_name)

            # Only keep COCO Search 18 classes
            if label_idx in COCO_SEARCH18_CLASSES:
                result['boxes'].append(box.tolist())
                result['scores'].append(float(score))
                result['labels'].append(label_idx)
                result['label_names'].append(label_name)

                # Match normalized categories (FIXED)
                if target_norm and label_norm == target_norm:
                    target_scores.append(float(score))

        if target_scores:
            result['target_category_score'] = max(target_scores)

    return result

def main():
    parser = argparse.ArgumentParser(description='Run Mask R-CNN on COCO Search 18 images')
    parser.add_argument('--annotations', default=ANNOTATIONS_FILE)
    parser.add_argument('--images', default=IMAGES_ROOT)
    parser.add_argument('--output', default=OUTPUT_FILE)
    parser.add_argument('--num-samples', type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Running Mask R-CNN on COCO Search 18")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    model = load_model(DEVICE)
    annotations = load_annotations(args.annotations)

    if args.num_samples:
        annotations = annotations[:args.num_samples]

    results = {}
    errors = []

    for annot in tqdm(annotations, desc="Processing"):
        img_name = annot['name']
        task = annot['task']
        bbox = annot['bbox']

        try:
            img_path = get_image_path(img_name, task, args.images)

            if not os.path.exists(img_path):
                errors.append(f"Missing: {img_path}")
                continue

            image = Image.open(img_path).convert('RGB')
            
            # Crop to bbox
            cropped = crop_bbox(image, bbox)

            detections = run_rcnn_on_image(cropped, model, DEVICE, target_category=task)

            # FIXED: prevent overwrite
            key = f"{img_name}_{task}"

            results[key] = {
                'task': task,
                'original_bbox': bbox,
                'detections': detections,
                'num_detections': len(detections['boxes']),
            }

        except Exception as e:
            errors.append(f"{img_name}: {str(e)}")

    if errors:
        print(f"\n⚠️ {len(errors)} errors (showing first 10):")
        for err in errors[:10]:
            print(" -", err)

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done!")

if __name__ == '__main__':
    main()