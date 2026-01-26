"""
full_pipeline_compute_and_aggregate_with_seg.py

Full pipeline:
 - Walks `OUTPUT_ROOT/output/<category>/<imageid>/<blurtype>/`
 - For each imageid + blurtype: compute metrics for rep0..rep9 (skip collages)
 - For segmentation blurtypes, generate masks on the fly using COCO annotations
 - Save per-image JSONs containing lists of metric values
 - Aggregate per-blurtype into Excel workbooks with 5 sheets (mean, median, min, max, std)
 - Save correlation CSVs (per-image and per-task) between metrics and nfix (if available)

CONFIGURE PATHS BELOW BEFORE RUNNING.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from ultralytics import YOLO
from scipy.stats import pearsonr
import warnings
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from metrics_helpers import *

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OUTPUT_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS/contextual")
DATASET_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/COCO_IMAGES")
PER_IMAGE_JSON_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS_PER_IMAGE_METRICS/contextual") #full pipeline uses this, but we don't need to run this if we are just finding corrs
# AGG_OUTPUT_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/aggregated_metrics_NORMALIZED")

PER_IMAGE_JSON_ROOT.mkdir(parents=True, exist_ok=True)

# Prompt template mapping: folder name -> prompt template
PROMPT_TEMPLATES = {
    "minimal": lambda cat: cat,
    "contextual": lambda cat: f"a {cat}",
    "realistic": lambda cat: f"a realistic {cat}",
    "natural_setting": lambda cat: f"a {cat} in a natural setting",
    "photorealistic": lambda cat: f"a {cat}, photorealistic",
    "original_quality": lambda cat: f"Full HD, 4K, high quality, high resolution, photorealistic image of {cat}",
    "plausible": lambda cat: f"a plausible {cat}",
    "plausible_scene": lambda cat: f"a {cat} in a plausible scene",
    "plausible_realistic": lambda cat: f"a plausible and realistic {cat}",
    "plausible_setting": lambda cat: f"a {cat} in a plausible setting",
    "plausible_placement": lambda cat: f"a {cat} plausibly placed",
    "highly_plausible": lambda cat: f"a highly plausible {cat}",
}

# Determine prompt name from OUTPUT_ROOT
PROMPT_NAME = OUTPUT_ROOT.name
PROMPT_TEMPLATE = PROMPT_TEMPLATES.get(PROMPT_NAME, lambda cat: cat)
print(f"Using prompt template for: {PROMPT_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
TARGET_SIZE = (224,224)

STATS = ["mean","median","min","max","std"]

COCO_ANNOTATIONS = [
    "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_train2017.json",
    "/home/kshaltiel/cluster_test_embed/coco_annotations/instances_val2017.json"
]

# ----------------- LOAD COCO ANNOTATIONS -----------------
coco_objs = []
filename_to_id_maps = []

for annot_path in COCO_ANNOTATIONS:
    coco = COCO(annot_path)
    coco_objs.append(coco)
    mapping = {img["file_name"]: img["id"] for img in coco.loadImgs(coco.getImgIds())}
    filename_to_id_maps.append(mapping)

# ----------------- LOAD MODELS -----------------
# Loading models now (no lazy loading)

# Placeholders for heavy model objects; set by load_models()
alexnet_full = None
alex_low = None
alex_mid = None
alex_final = None
embed_transform = None
rcnn_model = None
COCO_CATEGORIES = None
print("Loading models (this may take a while)...")

# AlexNet variants

alexnet_full = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
alex_low = nn.Sequential(*list(alexnet_full.features)[:3]).to(device).eval()
alex_mid = nn.Sequential(*list(alexnet_full.features)[:10]).to(device).eval()
alex_final = nn.Sequential(
    alexnet_full.features,
    nn.Flatten(),
    *list(alexnet_full.classifier.children())[:-1]
).to(device).eval()

embed_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


@torch.no_grad()
def get_embeddings_batch(img_list, model):
    t_list = torch.stack([embed_transform(img) for img in img_list]).to(device)
    return model(t_list)


# Mask R-CNN
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_model = maskrcnn_resnet50_fpn(weights=weights).eval().to(device)
COCO_CATEGORIES = weights.meta["categories"]
COCO_NAME_TO_IDX = {name.lower(): idx for idx, name in enumerate(COCO_CATEGORIES)}


@torch.no_grad()
def get_rcnn_confidence_patch(patch: Image.Image, category_name: str) -> float:
    img_tensor = TF.to_tensor(patch).to(device).unsqueeze(0)
    pred = rcnn_model(img_tensor)[0]
    target_idx = COCO_NAME_TO_IDX.get(category_name.lower(), None)
    if target_idx is None:
        return 0.0
    scores = pred['scores'][pred['labels'] == target_idx].cpu()
    return float(scores.max()) if len(scores) > 0 else 0.0


# YOLO
YOLO_WEIGHTS = "yolo12n.pt"
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_names = yolo_model.names if hasattr(yolo_model, "names") else None


def get_yolo_confidence_patch(patch: Image.Image, category_name: str) -> float:
    arr = np.array(patch)
    results = yolo_model(arr, imgsz=640, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
    try:
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return 0.0
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
        target_name = category_name.lower()
        matched = []
        for cls_idx, conf in zip(classes, confs):
            cls_name = yolo_names.get(int(cls_idx), "").lower() if yolo_names else ""
            if cls_name == target_name:
                matched.append(conf)
        return float(max(matched)) if matched else 0.0
    except Exception:
        return 0.0


# CLIP
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()  # remove .half() for safety
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# Stable Diffusion components for LaRE (reuse the same model used for inpainting)
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
sd_vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device).eval()
sd_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device).eval()
sd_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")


@torch.no_grad()
def get_clip_embedding(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    emb = clip.get_image_features(**inputs)
    return emb.squeeze(0).cpu()


@torch.no_grad()
def clip_object_presence_score(img: Image.Image, category_name: str):
    """
    Calculate object presence score using CLIP zero-shot classification.
    Returns:
        - presence_prob: probability that object is present
        - presence_logit_diff: logit difference (with - without)
    """
    text_prompts = [
        f"a photo containing a {category_name}",
        f"a photo without a {category_name}"
    ]
    
    inputs = clip_processor(
        text=text_prompts,
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    outputs = clip(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    presence_prob = probs[0, 0].item()
    logit_diff = logits_per_image[0, 0].item() - logits_per_image[0, 1].item()
    
    return presence_prob, logit_diff


@torch.no_grad()
def clip_text_similarity(img: Image.Image, category_name: str):
    """
    Compare image similarity to text "{category}" directly.
    Returns cosine similarity score.
    """
    text_prompt = f"a {category_name}"
    
    inputs = clip_processor(
        text=[text_prompt],
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    image_features = clip.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = clip.get_text_features(input_ids=inputs['input_ids'])
    
    # Compute cosine similarity: (A·B) / (||A|| * ||B||)
    dot_product = (image_features @ text_features.T)
    image_norm = image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features.norm(dim=-1, keepdim=True)
    
    similarity = (dot_product / (image_norm * text_norm.T)).item()
    
    return similarity


@torch.no_grad()
def calculate_lare(img: Image.Image, e: int = 5, t: int = 50, seed: int = 42):
    """
    Calculate Latent Reconstruction Error (LaRE) using diffusion model.
    
    LaRE measures how well the diffusion model can reconstruct an image by:
    1. Encoding image to latent space using VAE
    2. Adding noise at timestep t
    3. Using UNet to predict the noise
    4. Calculating L_ε = ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)
    5. Computing LaRE = (1/e) * Σ(L_ε ⊙ L_ε) (element-wise square, then sum)
    
    Args:
        img: Input image
        e: Number of Monte Carlo samples to average over (default=5)
           More samples = more stable estimate but slower
        t: Timestep for noise level (default=50)
           Lower t = less noise, higher t = more noise
           Range: 0-999 in DDPM, 50 is early in denoising (moderate noise)
        seed: Random seed for reproducibility (default=42)
           Same seed ensures all images use the same noise patterns
           This makes LaRE comparable across images
    
    Returns:
        LaRE score (float) - lower is better (easier to reconstruct)
    
    Implementation details:
    - Normalize to [-1, 1]: SD models expect this range
    - Scale by VAE scaling factor: Retrieved from model config
    - The formula follows the paper exactly: L_ε = true_noise - predicted_noise
    - Seed is set once per image, then e different noise samples are drawn
    """
    # Set seed for reproducibility - all images get same noise patterns
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Prepare image: resize to 512x512 (SD's native resolution)
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_tensor = TF.resize(img_tensor, (512, 512))
    
    # Normalize to [-1, 1]: SD models are trained on this range
    img_tensor = (img_tensor * 2.0) - 1.0
    
    # Encode to latent space using VAE
    latent = sd_vae.encode(img_tensor).latent_dist.sample()
    
    # Scale by VAE's latent scaling factor (typically 0.18215 for SD)
    # This is stored in the VAE config and ensures latents are in the expected range
    latent = latent * sd_vae.config.scaling_factor
    
    lare_sum = 0.0
    
    # Monte Carlo estimation: average over e noise samples
    # Each image gets the SAME e noise patterns (due to seed above)
    # This ensures fair comparison across different images
    for _ in range(e):
        # Sample random noise ε ~ N(0, I)
        noise = torch.randn_like(latent)
        
        # Create noisy latent: √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
        # This is the forward diffusion process at timestep t
        timesteps = torch.tensor([t], device=device)
        noisy_latent = sd_scheduler.add_noise(latent, noise, timesteps)
        
        # Predict noise using UNet: ε_θ(noisy_latent, t)
        # UNet requires text embeddings (encoder_hidden_states) even for unconditional generation
        # Using zeros for simplicity - technically should use CLIP text encoder("") but zeros
        # gives consistent results across all images which is what we need for fair comparison
        encoder_hidden_states = torch.zeros((1, 77, 768), device=device)  # (batch=1, tokens=77, dim=768)
        noise_pred = sd_unet(noisy_latent, timesteps, encoder_hidden_states).sample
        
        # Calculate latent error: L_ε = ε - ε_θ(...)
        # This is the reconstruction error in noise space
        latent_error = noise - noise_pred
        
        # Element-wise square: L_ε ⊙ L_ε
        squared_error = latent_error * latent_error
        
        # Sum all elements to get total error for this sample
        lare_sum += squared_error.sum().item()
    
    # Average over e Monte Carlo samples
    # Lower LaRE = model can easily predict the noise = high quality image
    lare = lare_sum / e
    
    return lare


# DINOv3
DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()  # remove .half()


@torch.no_grad()
def get_dino_embedding(img: Image.Image):
    inputs = dino_processor(images=img, return_tensors="pt").to(device)
    outputs = dino(**inputs)
    emb = outputs.pooler_output
    return emb.squeeze(0).cpu()

print('All models loaded.')

# ----------------- HELPER FUNCTIONS -----------------
def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def save_bbox_overlay(img: Image.Image, bbox, save_path: Path):
    """
    Draws a red rectangle for the bbox on the image and saves it.
    bbox: [x, y, width, height]
    """
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    overlay.save(save_path)


def generate_segmentation_mask(original_full: Image.Image, image_name: str, category_name: str):
    """
    Generates a binary mask for the given image and category using COCO annotations.
    Resizes the mask to match the size of `original_full`.
    """
    mask = Image.new("L", original_full.size, 0)
    target_w, target_h = original_full.size

    for coco, mapping in zip(coco_objs, filename_to_id_maps):
        img_id = mapping.get(image_name)
        if img_id is None:
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco.getCatIds(catNms=[category_name]))
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            rle = maskUtils.frPyObjects(ann["segmentation"], coco.imgs[img_id]['height'], coco.imgs[img_id]['width'])
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = np.sum(m, axis=2)
            m = np.clip(m, 0, 1)
            m_img = Image.fromarray((m*255).astype(np.uint8)).resize((target_w, target_h), Image.NEAREST)
            mask = Image.composite(Image.new("L", (target_w, target_h), 255), mask, m_img)

    return mask



def crop_by_bbox(original_full: Image.Image, bbox):
    x, y, w, h = bbox
    return original_full.crop((x, y, x+w, y+h))

def minmax_norm(t):
    return (t - t.min()) / (t.max() - t.min())

# ----------------- METRICS COMPUTATION -----------------
# ----------------- METRICS COMPUTATION -----------------
def process_image_pair(orig_crop: Image.Image, gen_img: Image.Image, category: str, mask_crop: Image.Image = None):
    """
    Compute metrics between orig_crop and gen_img.
    Returns a dict of metric_name -> scalar value.
    Includes pixel, AlexNet, CLIP, DINO, Mask R-CNN, and YOLO metrics.
    mask_crop: binary mask of the object (or bbox for non-segmented) used for area scaling
    """
    orig_r = orig_crop.resize(TARGET_SIZE, Image.LANCZOS)
    gen_r = gen_img.resize(TARGET_SIZE, Image.LANCZOS)

    # --- Pixel-level metrics ---
    pix_orig = torch.tensor(np.array(orig_r)).float().to(device)
    pix_gen = torch.tensor(np.array(gen_r)).float().to(device)

    metrics = {}
    metrics["pixel_L2"] = l2_flattened(pix_orig, pix_gen)
    metrics["pixel_cosine"] = cosine_flattened(pix_orig, pix_gen)

    # --- AlexNet features ---
    with torch.no_grad():
        a_low_o = alex_low(embed_transform(orig_r).unsqueeze(0).to(device))[0].cpu()
        a_low_g = alex_low(embed_transform(gen_r).unsqueeze(0).to(device))[0].cpu()
        a_mid_o = alex_mid(embed_transform(orig_r).unsqueeze(0).to(device))[0].cpu()
        a_mid_g = alex_mid(embed_transform(gen_r).unsqueeze(0).to(device))[0].cpu()
        feat_o = alexnet_full.features(embed_transform(orig_r).unsqueeze(0).to(device))
        feat_g = alexnet_full.features(embed_transform(gen_r).unsqueeze(0).to(device))
        feat_o_flat = feat_o.flatten(1).cpu()
        feat_g_flat = feat_g.flatten(1).cpu()
        classifier_features = nn.Sequential(*list(alexnet_full.classifier.children())[:-1]).to(device)
        final_o = classifier_features(feat_o_flat.to(device)).cpu()
        final_g = classifier_features(feat_g_flat.to(device)).cpu()

    metrics["low_L2"] = float(torch.norm(a_low_o.flatten() - a_low_g.flatten()).item())
    metrics["low_cosine"] = float(F.cosine_similarity(a_low_o.flatten().unsqueeze(0),
                                                       a_low_g.flatten().unsqueeze(0)).item())
    metrics["mid_L2"] = float(torch.norm(a_mid_o.flatten() - a_mid_g.flatten()).item())
    metrics["mid_cosine"] = float(F.cosine_similarity(a_mid_o.flatten().unsqueeze(0),
                                                       a_mid_g.flatten().unsqueeze(0)).item())
    metrics["high_L2"] = float(torch.norm(final_o.flatten() - final_g.flatten()).item())
    metrics["high_cosine"] = float(F.cosine_similarity(final_o.flatten().unsqueeze(0),
                                                        final_g.flatten().unsqueeze(0)).item())

    # --- Compute area fraction for mask scaling ---
    if mask_crop is None:
        mask_crop = Image.new("L", orig_crop.size, 255)  # full bbox
    mask_arr = np.array(mask_crop) > 0
    area_frac = mask_arr.sum() / (mask_crop.width * mask_crop.height)


    # --- CLIP features ---
    clip_o = get_clip_embedding(orig_r)
    clip_g = get_clip_embedding(gen_r)
    
    clip_o_norm = minmax_norm(clip_o)
    clip_g_norm = minmax_norm(clip_g)
    clip_cos = float(F.cosine_similarity(clip_o_norm.unsqueeze(0), clip_g_norm.unsqueeze(0)).item())
    metrics["clip_cosine"] = clip_cos

    # --- DINO features ---
    dino_o = get_dino_embedding(orig_r)
    dino_g = get_dino_embedding(gen_r)
    dino_o_norm = minmax_norm(dino_o)
    dino_g_norm = minmax_norm(dino_g)
    dino_cos = float(F.cosine_similarity(dino_o_norm.unsqueeze(0), dino_g_norm.unsqueeze(0)).item())
    metrics["dino_cosine"] = dino_cos

    # --- L2 distances for reference ---
    metrics["clip_L2"] = float(torch.norm(clip_o - clip_g).item())
    metrics["dino_L2"] = float(torch.norm(dino_o - dino_g).item())

    # --- Mask R-CNN and YOLO confidence ---
    metrics["rcnn_confidence"] = get_rcnn_confidence_patch(gen_img, category)
    metrics["yolo_confidence"] = get_yolo_confidence_patch(gen_img, category)

    # --- CLIP object presence scores ---
    gen_presence_prob, gen_logit_diff = clip_object_presence_score(gen_r, category)
    metrics["clip_presence_prob"] = gen_presence_prob
    metrics["clip_presence_logit_diff"] = gen_logit_diff
    
    gen_text_sim = clip_text_similarity(gen_r, category)
    metrics["clip_text_similarity"] = gen_text_sim

    # --- Latent Reconstruction Error (LaRE) ---
    metrics["lare"] = calculate_lare(gen_img)

    return metrics



# ----------------- FULL PIPELINE -------------------
def full_pipeline():
    SANITY_ROOT = Path("/home/kshaltiel/code/CSE-495-Code/sanity_check")
    SANITY_ROOT.mkdir(parents=True, exist_ok=True)
    sanity_imageids = set()

    for category_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name
        print(f"Processing category: {category_name}")

        for imageid_dir in category_dir.iterdir():
            if not imageid_dir.is_dir():
                continue
            imageid = imageid_dir.name
            # Only do sanity check for first 15 unique image IDs
            do_sanity = len(sanity_imageids) < 15 and imageid not in sanity_imageids
            if do_sanity:
                sanity_imageids.add(imageid)

            for blurtype_dir in imageid_dir.iterdir():
                if not blurtype_dir.is_dir():
                    continue
                blurtype = blurtype_dir.name
                segmented = "segmentation" in blurtype.lower()
                rep_metrics = defaultdict(list)

                for img_file in sorted(blurtype_dir.glob("*.jpg")):
                    if "collage" in img_file.name:
                        continue

                    gen_img = load_image_rgb(img_file)
                    orig_img_path = DATASET_ROOT / category_name / f"{imageid}.jpg"
                    orig_img = load_image_rgb(orig_img_path)

                    # --- Determine bbox ---
                    bbox = None
                    for coco, mapping in zip(coco_objs, filename_to_id_maps):
                        img_id = mapping.get(f"{imageid}.jpg")
                        if img_id is None:
                            continue
                        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco.getCatIds(catNms=[category_name]))
                        anns = coco.loadAnns(ann_ids)
                        if len(anns) == 0:
                            continue
                        ann = anns[0]
                        bbox = ann["bbox"]  # x, y, w, h
                        break
                    if bbox is None:
                        bbox = [0, 0, orig_img.width, orig_img.height]

                    # Crop original and generated images by bbox
                    orig_crop = crop_by_bbox(orig_img, bbox)
                    gen_crop = crop_by_bbox(gen_img, bbox)


                    # --- Apply segmentation mask if needed ---
                    if segmented:
                        mask = generate_segmentation_mask(orig_img, f"{imageid}.jpg", category_name)
                        mask_crop = crop_by_bbox(mask, bbox)
                        mask_crop = mask_crop.resize(TARGET_SIZE, Image.LANCZOS)
                        orig_crop = orig_crop.resize(TARGET_SIZE, Image.LANCZOS)
                        gen_crop = gen_crop.resize(TARGET_SIZE, Image.LANCZOS)

                        # Zero out background
                        gen_crop = Image.composite(gen_crop, Image.new("RGB", gen_crop.size, 0), mask_crop)
                        orig_crop = Image.composite(orig_crop, Image.new("RGB", orig_crop.size, 0), mask_crop)
                    else:
                        orig_crop = orig_crop.resize(TARGET_SIZE, Image.LANCZOS)
                        gen_crop = gen_crop.resize(TARGET_SIZE, Image.LANCZOS)
                        mask_crop = Image.new("L", orig_crop.size, 255)  # <--- put it here


                    # --- SANITY CHECK ---
                    # --- Compute metrics ---
                    # metrics = process_image_pair(orig_crop, gen_crop, category_name)
                    metrics = process_image_pair(orig_crop, gen_crop, category_name, mask_crop if segmented else orig_crop)
                    for k,v in metrics.items():
                        rep_metrics[k].append(v)

                # Save per-image JSON
                save_path = PER_IMAGE_JSON_ROOT / f"{category_name}_{imageid}_{blurtype}.json"
                with open(save_path, "w") as f:
                    json.dump(rep_metrics, f)

    print("Pipeline finished.")


# ----------------- AGGREGATION & CORRELATION -----------------
def aggregate_and_compute_correlation():
    # Deprecated wrapper
    print("aggregate_and_compute_correlation() is deprecated. Use generate_category_and_image_sets() instead.")


def _minmax_list(vals):
    """Safely min-max normalize a list-like numeric sequence. Returns list of floats.
    If constant or empty, returns the original values (or zeros for constant arrays)."""
    try:
        arr = np.array(vals, dtype=float)
    except Exception:
        return vals
    if arr.size == 0:
        return []
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        return [0.0 for _ in arr.tolist()]
    norm = (arr - amin) / (amax - amin)
    return norm.tolist()


def generate_category_and_image_sets(source_per_img_dir: Path = Path("./per_img")):
    """Read per-image JSONs from `source_per_img_dir` and produce two sets:
    - Unnormalized: writes per-image JSONs and per-blurtype category x metric CSVs to
      `per_image_metrics/` and `aggregated_metrics/`.
    - Normalized: same structure but metric rep-lists min-max normalized per-image and
      written to `per_image_metrics_NORMALIZED/` and `aggregated_metrics_NORMALIZED/`.

    This function only writes per-image JSONs (same format as input) and category x metric
    CSVs. It does NOT produce Excel workbooks or correlation files.
    """
    src = Path(source_per_img_dir)
    if not src.exists():
        print(f"Source per-image directory not found: {src}")
        return

    # output dirs
    out_perimg = Path("./per_image_metrics")
    out_perimg_norm = Pa
    metrics_9 = [
        "pixel_L2",
        "low_L2",
        "mid_L2",
        "high_L2",
        "pixel_cosine",
        "low_cosine",
        "mid_cosine",
        "high_cosine",
        "clip_cosine",
        "dino_cosine",
        "clip_L2",
        "dino_L2",
        "rcnn_confidence",
        "yolo_confidence",
        "clip_presence_prob",
        "clip_presence_logit_diff",
        "clip_text_similarity",
        "lare"
    ]

    def build_bucket(json_files, normalize=False):
        blurtype_to_cat_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for js in json_files:
            parts = js.stem.split("_")
            if len(parts) < 3:
                continue
            category_name = parts[0]
            imageid = parts[1]
            blurtype = "_".join(parts[2:])
            with open(js, "r") as f:
                metrics = json.load(f)
            if normalize:
                metrics = {k: _minmax_list(v) for k, v in metrics.items()}
            for k, v in metrics.items():
                blurtype_to_cat_metrics[blurtype][category_name][k].extend(v)
        return blurtype_to_cat_metrics

    # Build and write for both modes
    for normalize_flag, out_perimg_dir, out_agg_dir in [
        (False, out_perimg, out_agg),
        (True, out_perimg_norm, out_agg_norm),
    ]:
        # Write per-image JSONs
        for js in all_jsons:
            parts = js.stem.split("_")
            if len(parts) < 3:
                continue
            with open(js, "r") as f:
                metrics = json.load(f)
            metrics_out = {k: _minmax_list(v) for k, v in metrics.items()} if normalize_flag else metrics
            dst = out_perimg_dir / js.name
            with open(dst, "w") as f:
                json.dump(metrics_out, f)

        # Aggregate into category x metric CSVs
        bucket = build_bucket(all_jsons, normalize=normalize_flag)
        for blurtype, cat_dict in bucket.items():
            categories = sorted(cat_dict.keys())
            bycat_df = pd.DataFrame(index=categories, columns=metrics_9, dtype=float)
            for cat in categories:
                for metric_name in metrics_9:
                    vals = cat_dict[cat].get(metric_name, [])
                    stats = compute_stats_from_list(vals)
                    bycat_df.at[cat, metric_name] = stats.get("mean", np.nan)
            out_path = out_agg_dir / f"{blurtype}_metrics_by_category.csv"
            bycat_df.to_csv(out_path)

    print("Generated per-image and category x metric CSVs (normalized and unnormalized).")

# ----------------- RUN FULL PIPELINE -----------------
if __name__ == "__main__":
    # Always run the full pipeline first (compute per-image metrics), then aggregate both
    # the unnormalized and per-image minmax-normalized outputs.
    full_pipeline()
    generate_category_and_image_sets(PER_IMAGE_JSON_ROOT)
