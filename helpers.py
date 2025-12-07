from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import json
import numpy as np
from pycocotools import mask as maskUtils


#varibles 
TARGET_SIZE = (512, 512)
EMBED_SIZE  = (224, 224)

# load image from file
def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

#
def get_img_id_by_filename(filename_to_id, file_name):
    return filename_to_id.get(file_name, None)

# make sure directory exists -- if not, create it
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# scale the bbox to the image size 
def scale_bbox_xywh_to(dst_size, bbox_xywh, src_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    x, y, w, h = bbox_xywh
    sx = dst_w / src_w
    sy = dst_h / src_h
    x2 = x + w
    y2 = y + h
    x_s, y_s = int(x * sx), int(y * sy)
    x2_s, y2_s = int(x2 * sx), int(y2 * sy)
    return [x_s, y_s, x2_s - x_s, y2_s - y_s]

# crop the image to the bbox
def crop_bbox(img: Image.Image, bbox_xywh):
    x, y, w, h = bbox_xywh
    return img.crop((x, y, x + w, y + h))

# blur the bbox (gaussian blur)
def blur_bbox(image: Image.Image, bbox_xywh):
    """Apply Gaussian blur only to the bbox area and return mask."""
    x, y, w, h = bbox_xywh
    blurred_region = image.crop((x, y, x+w, y+h)).filter(ImageFilter.GaussianBlur(radius=10))
    img = image.copy()
    img.paste(blurred_region, (x, y))
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (x, y, x+w, y+h))
    return img, mask

# blur the mask 
def blur_mask(img: Image.Image, mask: Image.Image):
    """Return img where masked region is blurred."""
    img_copy = img.copy()
    blurred = img_copy.filter(ImageFilter.GaussianBlur(radius=10))
    mask_np = np.array(mask)
    mask_3c = np.stack([mask_np > 0] * 3, axis=2)
    img_arr = np.array(img_copy)
    blur_arr = np.array(blurred)
    img_arr[mask_3c] = blur_arr[mask_3c]
    return Image.fromarray(img_arr), mask

def make_collage(original, blurred, mask, inpainted, size=TARGET_SIZE):
    original = original.resize(size, Image.LANCZOS)
    blurred = blurred.resize(size, Image.LANCZOS)
    mask = mask.resize(size, Image.LANCZOS)
    inpainted = inpainted.resize(size, Image.LANCZOS)

    w, h = size
    collage = Image.new("RGB", (w*2, h*2))
    collage.paste(original, (0, 0))
    collage.paste(blurred, (w, 0))
    collage.paste(mask.convert("RGB"), (0, h))
    collage.paste(inpainted, (w, h))
    return collage

def load_bbox_json(json_path):
    # -------------------- Load bbox JSON --------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    image_to_bboxes = {}
    for item in data:
        image_to_bboxes.setdefault(item["name"], {})[item.get("task","").lower()] = item["bbox"]

    return image_to_bboxes

def load_coco_annotations(coco_json_path, HAS_COCO, COCO):

    coco = None
    filename_to_id = {}
    if HAS_COCO and Path(coco_json_path).exists():
        try:
            coco = COCO(coco_json_path)
            filename_to_id = {meta["file_name"]: img_id for img_id, meta in coco.imgs.items()}
            print(f"Loaded COCO instances from {coco_json_path}")
        except Exception as e:
            print(f"Failed to load COCO from {coco_json_path}: {e}")
            coco = None

    return coco, filename_to_id

def decode_segmentation_to_numpy(ann, coco):
    """
    Decode ann['segmentation'] into a numpy mask (H_orig, W_orig) with 0/1 values.
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return None, None, None
    img_info = coco.loadImgs(ann["image_id"])[0]
    H_orig, W_orig = img_info["height"], img_info["width"]

    if isinstance(seg, list):
        # polygon(s): use frPyObjects -> decode
        rles = maskUtils.frPyObjects(seg, H_orig, W_orig)
        rle = maskUtils.merge(rles)
        mask_np = maskUtils.decode(rle)
    elif isinstance(seg, dict):
        mask_np = maskUtils.decode(seg)
    else:
        return None, W_orig, H_orig

    mask_np = np.asarray(mask_np)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_bin = (mask_np > 0).astype(np.uint8)
    return mask_bin, W_orig, H_orig

def get_segmentation_mask_resized(image_full: Image.Image, ann, coco=None) -> Image.Image:
    """
    Return a PIL 'L' mask sized to image_full.size where 255 indicates object.
    Uses letterbox resizing so mask aligns even if Search18 images were letterboxed.
    If coco provided and ann has segmentation, decode at original COCO size and letterbox-resize into image_full.size.
    Otherwise falls back to scaled bbox with the same letterbox transform.
    """
    # Try segmentation first (COCO)
    if coco is not None and "segmentation" in ann and ann["segmentation"]:
        try:
            mask_np, W_orig, H_orig = decode_segmentation_to_numpy(ann, coco)
            if mask_np is not None:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
                mask_letterbox, sx, sy, dx, dy = resize_with_letterbox(mask_pil, W_orig, H_orig, image_full.width, image_full.height)
                return mask_letterbox
        except Exception as e:
            print(f"⚠️ Failed to decode segmentation: {e}")

    raise ValueError("Annotation contains neither segmentation nor bbox.")

def resize_with_letterbox(mask_pil: Image.Image, orig_w: int, orig_h: int, new_w: int, new_h: int) -> (Image.Image, float, float, int, int):
    """
    Resize mask (orig_w,orig_h) -> (new_w,new_h) preserving aspect ratio
    with letterbox (padding). Returns: resized_mask, scale, sx, dx, dy
    scale = scale used (uniform), sx, sy same = scale, dx,dy = top-left offset where resized content was pasted.
    """
    # compute scale preserving aspect ratio
    scale = min(new_w / orig_w, new_h / orig_h)
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    mask_resized = mask_pil.resize((resized_w, resized_h), resample=Image.NEAREST)
    # create letterbox canvas
    canvas = Image.new("L", (new_w, new_h), 0)
    dx = (new_w - resized_w) // 2
    dy = (new_h - resized_h) // 2
    canvas.paste(mask_resized, (dx, dy))
    # ensure binary
    canvas = canvas.point(lambda p: 255 if p>128 else 0).convert("L")
    return canvas, scale, scale, dx, dy

def scale_bbox_with_letterbox(bbox_orig, orig_w, orig_h, new_w, new_h):
    """Scale bbox [x,y,w,h] from original coords into letterboxed new image coords."""
    x, y, w, h = bbox_orig
    # compute scale and offsets
    scale = min(new_w / orig_w, new_h / orig_h)
    sx = scale; sy = scale
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    dx = (new_w - resized_w) // 2
    dy = (new_h - resized_h) // 2
    x_new = x * sx + dx
    y_new = y * sy + dy
    w_new = w * sx
    h_new = h * sy
    return [x_new, y_new, w_new, h_new]
