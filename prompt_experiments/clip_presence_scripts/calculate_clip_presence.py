#!/usr/bin/env python3
"""
Calculate CLIP-based object presence scores for existing generated images
Tests zero-shot classification approach: "image with {category}" vs "image without {category}"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import pandas as pd
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Paths --------------------
# Start with one prompt to test
PROMPT_NAME = "minimal"  # We'll test on minimal first
output_root = Path(f"/home/kshaltiel/code/CSE-495-Code/output/PROMPT_EXPERIMENTS/{PROMPT_NAME}")
results_dir = Path("/home/kshaltiel/code/CSE-495-Code/output/CLIP_PRESENCE_SCORES")
results_dir.mkdir(parents=True, exist_ok=True)

# -------------------- Model Loading --------------------
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("✓ CLIP model loaded")

# -------------------- Helper Functions --------------------
def load_image(path):
    return Image.open(path).convert("RGB")

def clip_object_presence_score(img, category_name):
    """
    Calculate object presence score using CLIP zero-shot classification
    Returns:
        - presence_prob: probability that object is present
        - presence_logit_diff: logit difference (with - without)
    """
    # Create text prompts
    text_prompts = [
        f"a photo containing a {category_name}",
        f"a photo without a {category_name}"
    ]
    
    # Process inputs
    inputs = clip_processor(
        text=text_prompts,
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Get outputs
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, 2]
        probs = logits_per_image.softmax(dim=1)  # shape: [1, 2]
    
    presence_prob = probs[0, 0].item()  # probability of "with" category
    logit_diff = logits_per_image[0, 0].item() - logits_per_image[0, 1].item()
    
    return presence_prob, logit_diff

def clip_object_presence_score_v2(img, category_name):
    """
    Alternative: Compare image similarity to text "{category}" directly
    Returns cosine similarity score
    """
    text_prompt = f"a {category_name}"
    
    inputs = clip_processor(
        text=[text_prompt],
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = clip_model.get_text_features(input_ids=inputs['input_ids'])
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (image_features @ text_features.T).item()
    
    return similarity

def sanitize_category(cat):
    return cat.replace(' ', '_').replace('/', '_')

# -------------------- Main --------------------
print(f"Starting CLIP presence scoring for {PROMPT_NAME} prompt...")

all_results = []

# Process each category
categories = [d for d in output_root.iterdir() if d.is_dir()]
print(f"Found {len(categories)} categories")

for category_dir in tqdm(categories, desc="Categories"):
    category_name = category_dir.name
    category_sanitized = sanitize_category(category_name)
    
    # Process each image folder
    img_folders = [d for d in category_dir.iterdir() if d.is_dir()]
    
    for img_folder in tqdm(img_folders, desc=f"  {category_name}", leave=False):
        img_base = img_folder.name
        
        # Load original image
        original_path = img_folder / f"{img_base}_original.jpg"
        if not original_path.exists():
            continue
        
        original_img = load_image(original_path)
        
        # Score original image
        orig_presence_prob, orig_logit_diff = clip_object_presence_score(original_img, category_name)
        orig_similarity = clip_object_presence_score_v2(original_img, category_name)
        
        # Process bbox and segmentation conditions
        for condition in ["bbox", "segmentation"]:
            condition_dir = img_folder / condition
            if not condition_dir.exists():
                continue
            
            rep_files = sorted(condition_dir.glob(f"{img_base}_rep*.jpg"))
            rep_files = [f for f in rep_files if "collage" not in f.name]
            
            if len(rep_files) == 0:
                continue
            
            # Process each rep
            for rep_file in rep_files:
                rep_num = int(rep_file.stem.split("_rep")[-1])
                
                inpainted_img = load_image(rep_file)
                
                # Calculate CLIP presence scores
                inp_presence_prob, inp_logit_diff = clip_object_presence_score(inpainted_img, category_name)
                inp_similarity = clip_object_presence_score_v2(inpainted_img, category_name)
                
                result = {
                    "prompt": PROMPT_NAME,
                    "category": category_name,
                    "image": img_base,
                    "condition": condition,
                    "rep": rep_num,
                    # Original scores
                    "original_presence_prob": orig_presence_prob,
                    "original_logit_diff": orig_logit_diff,
                    "original_text_similarity": orig_similarity,
                    # Inpainted scores
                    "inpainted_presence_prob": inp_presence_prob,
                    "inpainted_logit_diff": inp_logit_diff,
                    "inpainted_text_similarity": inp_similarity,
                    # Differences (original - inpainted)
                    "delta_presence_prob": orig_presence_prob - inp_presence_prob,
                    "delta_logit_diff": orig_logit_diff - inp_logit_diff,
                    "delta_text_similarity": orig_similarity - inp_similarity,
                }
                
                all_results.append(result)

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Save full results
csv_path = results_dir / f"{PROMPT_NAME}_clip_presence_scores.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved full results: {csv_path}")
print(f"   Total rows: {len(df)}")

# Compute summary statistics per condition
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for condition in ["bbox", "segmentation"]:
    condition_df = df[df["condition"] == condition]
    
    print(f"\n{condition.upper()}:")
    print(f"  Original images:")
    print(f"    Mean presence prob: {condition_df['original_presence_prob'].mean():.4f}")
    print(f"    Mean logit diff: {condition_df['original_logit_diff'].mean():.4f}")
    print(f"    Mean text similarity: {condition_df['original_text_similarity'].mean():.4f}")
    
    print(f"  Inpainted images:")
    print(f"    Mean presence prob: {condition_df['inpainted_presence_prob'].mean():.4f}")
    print(f"    Mean logit diff: {condition_df['inpainted_logit_diff'].mean():.4f}")
    print(f"    Mean text similarity: {condition_df['inpainted_text_similarity'].mean():.4f}")
    
    print(f"  Delta (original - inpainted):")
    print(f"    Mean delta presence prob: {condition_df['delta_presence_prob'].mean():.4f}")
    print(f"    Mean delta logit diff: {condition_df['delta_logit_diff'].mean():.4f}")
    print(f"    Mean delta text similarity: {condition_df['delta_text_similarity'].mean():.4f}")

# Per-category analysis
print("\n" + "="*80)
print("PER-CATEGORY ANALYSIS (Inpainted Presence Probability)")
print("="*80)

category_summary = df.groupby(['category', 'condition']).agg({
    'inpainted_presence_prob': ['mean', 'std'],
    'delta_presence_prob': ['mean', 'std']
}).round(4)

print(category_summary)

# Save summary
summary_path = results_dir / f"{PROMPT_NAME}_clip_presence_summary.csv"
category_summary.to_csv(summary_path)
print(f"\n✅ Saved summary: {summary_path}")

print(f"\n🎉 CLIP presence scoring complete for {PROMPT_NAME}!")
