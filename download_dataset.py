import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
import random

# ---------------- CONFIG ----------------
COCO_TRAIN_JSON = Path(r"C:\Users\khans\Desktop\coco_annotations\instances_train2017.json")
COCO_VAL_JSON = Path(r"C:\Users\khans\Desktop\coco_annotations\instances_val2017.json")
COCO_IMAGES_DIR = Path(r"C:\Users\khans\Desktop\CSE-495-CODE\COCO_IMAGES")
MAIN_IMAGE_FOLDER = Path(r"C:\Users\khans\Desktop\RESEARCH\coco_search18_images_TP_gpu\images")

CATEGORIES_OF_INTEREST = [
    "bottle","bowl","car","chair","clock","cup","fork","keyboard",
    "knife","laptop","microwave","mouse","oven","potted plant",
    "sink","stop sign","toilet","tv"
]
VISUALIZE_COUNT = 15

COCO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD COCO ANNOTATIONS ----------------
def load_coco_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

coco_train = load_coco_json(COCO_TRAIN_JSON)
coco_val = load_coco_json(COCO_VAL_JSON)

# Merge images and annotations
coco_data = {
    "images": coco_train["images"] + coco_val["images"],
    "annotations": coco_train["annotations"] + coco_val["annotations"],
    "categories": coco_train["categories"],  # categories are the same in train & val
}

cat_name_to_id = {c["name"]: c["id"] for c in coco_data["categories"]}
cat_ids_of_interest = [cat_name_to_id[c] for c in CATEGORIES_OF_INTEREST if c in cat_name_to_id]

# Map COCO image IDs to info
id_to_info = {img["id"]: img for img in coco_data["images"]}

print(f"Total images in COCO (train+val): {len(id_to_info)}")

# ---------------- FILTER IMAGES BY CATEGORY ----------------
category_to_images = {}

for cat_name in CATEGORIES_OF_INTEREST:
    cat_folder = MAIN_IMAGE_FOLDER / cat_name
    if not cat_folder.is_dir():
        continue

    images_in_category = []
    cat_id = cat_name_to_id[cat_name]

    for img_file in cat_folder.iterdir():
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img_id = int(img_file.stem)  # remove leading zeros
        img_info = id_to_info.get(img_id)
        if img_info is None:
            print(f"COCO info not found for file: {img_file.name} (ID: {img_id})")
            continue

        # Only include if the image has at least one annotation in COCO
        anns_for_image = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]]
        if not anns_for_image:
            continue

        images_in_category.append(img_info)

    if images_in_category:
        category_to_images[cat_name] = images_in_category
        print(f"Category {cat_name}: {len(images_in_category)} images found")

# ---------------- DOWNLOAD IMAGES ----------------
print("Downloading images by category...")
for cat_name, images in category_to_images.items():
    cat_save_folder = COCO_IMAGES_DIR / cat_name
    cat_save_folder.mkdir(parents=True, exist_ok=True)

    for img_meta in images:
        save_path = cat_save_folder / img_meta["file_name"]
        if save_path.exists():
            continue

        url = img_meta.get("coco_url") or img_meta.get("flickr_url")
        if not url:
            print(f"No URL for {img_meta['file_name']}")
            continue

        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {img_meta['file_name']} to {cat_save_folder}")
        except Exception as e:
            print(f"Failed to download {img_meta['file_name']}: {e}")

# ---------------- VISUALIZE SAMPLE ----------------
print(f"Visualizing {VISUALIZE_COUNT} images...")
all_images_for_visualization = [img for imgs in category_to_images.values() for img in imgs]
sample_images = random.sample(all_images_for_visualization, min(VISUALIZE_COUNT, len(all_images_for_visualization)))

# Map image_id -> annotations
imgid_to_anns = {}
for cat_imgs in category_to_images.values():
    for img_meta in cat_imgs:
        anns = [ann for ann in coco_data["annotations"]
                if ann["image_id"] == img_meta["id"] and ann["category_id"] in cat_ids_of_interest]
        if anns:
            imgid_to_anns[img_meta["id"]] = anns

for img_meta in sample_images:
    img_id = img_meta["id"]
    anns = imgid_to_anns.get(img_id, [])
    if not anns:
        continue

    first_cat_id = anns[0]["category_id"]
    cat_name = next(c["name"] for c in CATEGORIES_OF_INTEREST if cat_name_to_id[c] == first_cat_id)
    img_path = COCO_IMAGES_DIR / cat_name / img_meta["file_name"]
    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for ann in anns:
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0, 255), width=3)

        seg = ann.get("segmentation")
        if seg:
            if isinstance(seg, list):
                for poly in seg:
                    xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    draw.polygon(xy, fill=(0, 255, 0, 80))
            elif isinstance(seg, dict):
                mask = maskUtils.decode(seg)
                mask_img = Image.fromarray(mask * 80).convert("L")
                img.paste(Image.new("RGBA", img.size, (0, 255, 0, 80)), mask=mask_img)

    img.show()
