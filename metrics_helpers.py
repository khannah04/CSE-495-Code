import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from scipy.stats import pearsonr
import torch.nn.functional as F

TARGET_SIZE = (512, 512)
STATS = ["mean", "median", "min", "max", "std"]

# ----------------- UTILITIES -----------------
def load_image_rgb(path: Path):
    return Image.open(path).convert("RGB")

def l2_flattened(t1, t2):
    return float(torch.norm(t1.flatten() - t2.flatten()).item())

def cosine_flattened(t1, t2):
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    return float(F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item())

def safe_np_array(x):
    try:
        return np.array(x, dtype=float)
    except Exception:
        return None

def compute_stats_from_list(vals):
    arr = safe_np_array(vals)
    if arr is None or arr.size == 0:
        return {s: math.nan for s in STATS}
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr, ddof=0)),
    }

def pearson_safe(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.logical_or(np.isnan(x), np.isnan(y))
    x2 = x[mask]
    y2 = y[mask]
    if x2.size < 2:
        return (math.nan, math.nan)
    try:
        r, p = pearsonr(x2, y2)
        return float(r), float(p)
    except Exception:
        return (math.nan, math.nan)
    
def load_bbox_json(BBOX_JSON):
    # ----------------- LOAD BBOX JSON -----------------
    print("Loading bbox JSON...")
    with open(BBOX_JSON, "r") as f:
        bbox_data = json.load(f)

    image_to_bboxes = {}
    # Expecting entries with keys "name" (like 000000014044.jpg), "task" (category), "bbox" [x,y,w,h], and possibly "nfix".
    for item in bbox_data:
        name = item.get("name")
        if name is None:
            continue
        key = name  # '000000014044.jpg'
        d = image_to_bboxes.setdefault(key, {})
        task = item.get("task","").lower()
        d[task] = item.get("bbox")
        # store nfix if present as well
        if "nfix" in item:
            d.setdefault("_meta", {})["nfix"] = item.get("nfix")
        
    print(f"Loaded bbox entries: {len(image_to_bboxes)}")
    return image_to_bboxes
