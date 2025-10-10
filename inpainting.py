from pathlib import Path
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
from pycocotools import mask as maskUtils

try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except Exception:
    COCO = None
    HAS_COCO = False

from helpers import *

#TODO: change paths to match the cluster 
dataset_root = Path("/home/kshaltiel/cluster_test_embed/coco_search_18_TP/images")
output_root = Path("/home/kshaltiel/cluster_test_embed/images_blur_metrics")
coco_instances_json_train = "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_train2017.json"
coco_instances_json_test = "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_val2017.json"

output_root.mkdir(parents=True, exist_ok=True)

# --- load COCO annotations for train and test ---
coco_train, filename_to_id_train = load_coco_annotations(coco_instances_json_train, HAS_COCO, COCO)
coco_test, filename_to_id_test = load_coco_annotations(coco_instances_json_test, HAS_COCO, COCO)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    revision="fp16", #TODO: check to see what this does 
    torch_dtype=torch.float16,
).to("cuda")

# load the stable diffusion 1.5 inpainting model 
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting

for category_dir in dataset_root.iterdir():
    if not category_dir.is_dir():
        continue
    CATEGORY_NAME = category_dir.name
    print(f"Processing category: {CATEGORY_NAME}")

    for img_path in category_dir.iterdir(): #for each image in the category
        img_name = img_path.name
        base = img_path.stem
        out_dir = output_root / CATEGORY_NAME / base
        ensure_dir(out_dir)

        #load image 
        original_full = load_image(img_path)
        original_size = original_full.size
        
        # --- get COCO image ID from train first, then test if not found ---
        img_id = filename_to_id_train.get(img_name)
        coco_obj = coco_train
        if img_id is None:
            img_id = filename_to_id_test.get(img_name)
            coco_obj = coco_test

        if img_id is None:
            print(f"⚠️ COCO has no entry for {img_name}, skipping.")
            continue

        # --- get annotations for this image ---
        ann_ids = coco_obj.getAnnIds(imgIds=img_id)
        anns = coco_obj.loadAnns(ann_ids)
        if len(anns) == 0:
            print(f"⚠️ No annotations for {img_name} in COCO, skipping.")
            continue

        # find annotation with matching category
        ann = next(
            (a for a in anns if coco_obj.loadCats(a["category_id"])[0]["name"] == CATEGORY_NAME),
            None
        )

        if ann is None:
            print(f"⚠️ No matching {CATEGORY_NAME} annotation for {img_name}, skipped.")
            continue

        # ✅ use COCO bbox as canonical bbox
        bbox_orig = ann.get("bbox", None)
        if bbox_orig is None:
            print(f"⚠️ No bbox in COCO for {img_name}, skipped.")
            continue

        print(f"✔️ Found bbox for {img_name}: {bbox_orig}")

        # --- resize bbox to current image size ---
        img_info = coco_obj.loadImgs(ann["image_id"])[0]
        orig_w, orig_h = img_info["width"], img_info["height"]
        bbox_resized = scale_bbox_with_letterbox(ann["bbox"], orig_w, orig_h, original_full.width, original_full.height)

        # --- get segmentation mask, resized to current image ---
        try:
            seg_mask_resized = get_segmentation_mask_resized(original_full, ann, coco_obj)
        except Exception as e:
            print(f"⚠️ Failed to decode segmentation for {img_name}: {e}")
            seg_mask_resized = None

        # --- create a mask from the resized bbox ---
        bbox_mask_resized = Image.new("L", original_full.size, 0)
        x, y, w, h = map(int, bbox_resized)
        draw = ImageDraw.Draw(bbox_mask_resized)
        draw.rectangle([x, y, x + w, y + h], fill=255)

        # --- Resize everything to pipeline size ---
        original_512 = original_full.resize(TARGET_SIZE, Image.LANCZOS)
        bbox_mask_512 = bbox_mask_resized.resize(TARGET_SIZE, resample=Image.NEAREST)
        bbox_mask_512 = bbox_mask_512.point(lambda p: 255 if p > 128 else 0).convert("L")

        if seg_mask_resized:
            seg_mask_512 = seg_mask_resized.resize(TARGET_SIZE, resample=Image.NEAREST)
            seg_mask_512 = seg_mask_512.point(lambda p: 255 if p > 128 else 0).convert("L")
            seg_blurred_512, _ = blur_mask(original_512, seg_mask_512)
        else:
            seg_mask_512 = None
            seg_blurred_512 = None

        bbox_blurred_512, _ = blur_mask(original_512, bbox_mask_512)

        mask_conditions = [
            ("segmentation_no_blur", original_512, seg_mask_512),
            ("segmentation_blur", seg_blurred_512, seg_mask_512),
            ("bbox_no_blur", original_512, bbox_mask_512),
            ("bbox_blur", bbox_blurred_512, bbox_mask_512),
        ]

        for condition_name, img_input, mask_input in mask_conditions:
            if mask_input is None:
                continue  # skip if mask unavailable
            # TODO: what should the images (for testing) be? 

            for i in range(10): #generate 10 inpaints per image
                generator = torch.Generator(device="cuda").manual_seed(i+1)

                try: 
                    result = pipe(
                        prompt=f"Full HD, 4K, high quality, high resolution, photorealistic image of {CATEGORY_NAME}",
                        negative_prompt="bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality",
                        image = img_input, 
                        mask_image = mask_input,
                        generator=generator,
                        revision="fp16", #TODO: check to see what this does 
                        torch_dtype=torch.float16,
                    ).images[0]

                except Exception as e: 
                    print(f"  Inpaint error on {img_name}: {e}")
                    continue
                
                # Save inpainted result
                result.save(out_dir / f"{base}_{condition_name}_rep{i}.jpg")

                collage = make_collage(
                    original_512,
                    bbox_blurred_512,
                    bbox_mask_512,
                    result
                )
                collage.save(out_dir / f"{base}_{condition_name}_rep{i}_collage.jpg")

            # --- Save original ---
            original_512.save(out_dir / f"{base}_original.jpg")
