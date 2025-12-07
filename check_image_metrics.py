import json
from pathlib import Path

# Search for metrics files containing this image ID
metrics_dir = Path("/home/kshaltiel/code/CSE-495-Code/per_image_metrics")
image_id = "000000409338"
category = "oven"
rep_num = 4

matching_files = list(metrics_dir.glob(f"*{image_id}*.json"))

if matching_files:
    print(f"Found {len(matching_files)} matching files:")
    for f in matching_files:
        print(f"\n{f.name}")
        with open(f, 'r') as file:
            data = json.load(file)
            
        if 'yolo_confidence' in data:
            yolo_confs = data['yolo_confidence']
            print(f"YOLO confidences: {yolo_confs}")
            print(f"rep{rep_num} YOLO confidence: {yolo_confs[rep_num]}")
            
        if 'rcnn_confidence' in data:
            rcnn_confs = data['rcnn_confidence']
            print(f"\nRCNN confidences: {rcnn_confs}")
            print(f"rep{rep_num} RCNN confidence: {rcnn_confs[rep_num]}")
else:
    print(f"No metrics files found for image {image_id}")
