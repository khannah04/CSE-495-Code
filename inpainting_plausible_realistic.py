from pathlib import Path
import sys
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

# Add parent directory to path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import *

# --- Paths ---
dataset_root = Path("/home/kshaltiel/code/CSE-495-Code/COCO_IMAGES")
output_root = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS")
coco_instances_json_train = "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_train2017.json"
coco_instances_json_test = "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_val2017.json"

output_root.mkdir(parents=True, exist_ok=True)

# --- load COCO annotations ---
coco_train, filename_to_id_train = load_coco_annotations(coco_instances_json_train, HAS_COCO, COCO)
coco_test, filename_to_id_test = load_coco_annotations(coco_instances_json_test, HAS_COCO, COCO)

# --- Load inpainting pipeline ---
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

# --- Define prompt strategy: plausible_realistic ---
PROMPT_STRATEGIES = {
    "plausible_realistic": lambda cat: f"a plausible and realistic {cat}",
}

# --- Iterate over prompt strategies ---
for prompt_name, prompt_fn in PROMPT_STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"TESTING PROMPT STRATEGY: {prompt_name}")
    print(f"{'='*60}\n")

    # --- Iterate over categories ---
    for category_dir in dataset_root.iterdir():
        if not category_dir.is_dir():
            continue
        CATEGORY_NAME = category_dir.name
        print(f"Processing category: {CATEGORY_NAME}")

        for img_path in category_dir.iterdir():  # For each image
            img_name = img_path.name
            base = img_path.stem
            out_dir = output_root / prompt_name / CATEGORY_NAME / base
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load image
            original_full = load_image(img_path)
            original_size = original_full.size

            # --- get COCO image ID ---
            img_id = filename_to_id_train.get(img_name)
            coco_obj = coco_train
            if img_id is None:
                img_id = filename_to_id_test.get(img_name)
                coco_obj = coco_test

            if img_id is None:
                print(f"⚠️ COCO has no entry for {img_name}, skipping.")
                continue

            # --- get annotations ---
            ann_ids = coco_obj.getAnnIds(imgIds=img_id)
            anns = coco_obj.loadAnns(ann_ids)
            if len(anns) == 0:
                print(f"⚠️ No annotations for {img_name} in COCO, skipping.")
                continue

            ann = next(
                (a for a in anns if coco_obj.loadCats(a["category_id"])[0]["name"] == CATEGORY_NAME),
                None
            )

            if ann is None:
                print(f"⚠️ No matching {CATEGORY_NAME} annotation for {img_name}, skipped.")
                continue

            bbox_orig = ann.get("bbox", None)
            if bbox_orig is None:
                print(f"⚠️ No bbox in COCO for {img_name}, skipped.")
                continue

            print(f"✔️ Found bbox for {img_name}: {bbox_orig}")

            # --- Resize bbox to current image ---
            img_info = coco_obj.loadImgs(ann["image_id"])[0]
            orig_w, orig_h = img_info["width"], img_info["height"]
            bbox_resized = scale_bbox_with_letterbox(ann["bbox"], orig_w, orig_h, original_full.width, original_full.height)

            # --- Get segmentation mask ---
            try:
                seg_mask_resized = get_segmentation_mask_resized(original_full, ann, coco_obj)
            except Exception as e:
                print(f"⚠️ Failed to decode segmentation for {img_name}: {e}")
                seg_mask_resized = None

            # --- Create bbox mask ---
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

            # --- Define conditions (no blur variants) ---
            mask_conditions = [
                ("segmentation", original_512, seg_mask_512),
                ("bbox", original_512, bbox_mask_512),
            ]

            # --- Get prompt for this strategy ---
            prompt = prompt_fn(CATEGORY_NAME)
            print(f"  Using prompt: '{prompt}'")

            # --- Process each condition ---
            for condition_name, img_input, mask_input in mask_conditions:
                if mask_input is None:
                    continue

                # Create folder for this condition
                condition_dir = out_dir / condition_name
                condition_dir.mkdir(parents=True, exist_ok=True)

                for i in range(10):  # Generate 10 inpaints per image
                    generator = torch.Generator(device="cuda").manual_seed(i + 1)

                    try:
                        result = pipe(
                            prompt=prompt,
                            # NO negative prompt for maximum creativity
                            image=img_input,
                            mask_image=mask_input,
                            generator=generator,
                            torch_dtype=torch.float16,
                        ).images[0]

                    except Exception as e:
                        print(f"  Inpaint error on {img_name}: {e}")
                        continue

                    # Save inpainted result inside condition folder
                    result.save(condition_dir / f"{base}_rep{i}.jpg")

                    # --- Save collage showing the actual condition ---
                    if condition_name.startswith("bbox"):
                        condition_preview = bbox_blurred_512 if "blur" in condition_name else original_512
                        mask_preview = bbox_mask_512
                    elif condition_name.startswith("segmentation"):
                        condition_preview = seg_blurred_512 if "blur" in condition_name else original_512
                        mask_preview = seg_mask_512
                    else:
                        condition_preview = original_512
                        mask_preview = mask_input

                    collage = make_collage(
                        original_512,
                        condition_preview,
                        mask_preview,
                        result
                    )
                    collage.save(condition_dir / f"{base}_rep{i}_collage.jpg")

            # --- Save original image once per image ---
            original_512.save(out_dir / f"{base}_original.jpg")

print(f"\n{'='*60}")
print("PLAUSIBLE_REALISTIC PROMPT COMPLETE!")
print(f"{'='*60}")
