import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# ================= CONFIGURATION =================
ROOT = Path(__file__).resolve().parent   # sky_detection_pipeline
BASE_DIR = ROOT.parent                  

SRC_IMG_DIR = BASE_DIR / "gene_detection_merged_corrected" / "train" / "images"
SRC_LBL_DIR = BASE_DIR / "gene_detection_merged_corrected" / "train" / "labels"

OUTPUT_BASE_DIR = BASE_DIR / "FULL_DATASET" / "gene_dataset_5folds"
# =================================================

# 1. Group files by unique Cell ID (Filename + Cell Number)
groups = defaultdict(list)
all_images = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for f in all_images:
    prefix = ""
    cell_id = ""
    # Matches the prefix before and the cell number after the channel 
    for marker in ['_FITC_', '_ORANGE_', '_AQUA_']:
        if marker in f:
            parts = f.split(marker)
            prefix = parts[0] 
            cell_id = parts[1].split('.')[0] 
            break
    
    if prefix and cell_id:
        group_key = f"{prefix}_{cell_id}"
        groups[group_key].append(f)

unique_keys = list(groups.keys())

# 2. Shuffle with a fixed seed for reproducibility
random.seed(42)
random.shuffle(unique_keys)

# 3. Create 5-Fold Splits
num_folds = 5
avg = len(unique_keys) / float(num_folds)
split_keys = []
last = 0.0

while last < len(unique_keys):
    split_keys.append(unique_keys[int(last):int(last + avg)])
    last += avg

# 4. Execute file distribution
print(f"--- 5-Fold Grouped Split ---")
print(f"Total Unique Cells Found: {len(unique_keys)}\n")

for i, fold_keys in enumerate(split_keys):
    fold_dir = os.path.join(OUTPUT_BASE_DIR, f'fold_{i+1}')
    img_dist = os.path.join(fold_dir, 'images')
    lbl_dist = os.path.join(fold_dir, 'labels')
    
    os.makedirs(img_dist, exist_ok=True)
    os.makedirs(lbl_dist, exist_ok=True)

    img_count = 0
    lbl_count = 0

    for key in fold_keys:
        for img_file in groups[key]:
            # Copy Image
            shutil.copy(os.path.join(SRC_IMG_DIR, img_file), os.path.join(img_dist, img_file))
            img_count += 1
            
            # Copy corresponding Label
            base_name = os.path.splitext(img_file)[0]
            lbl_file = base_name + '.txt'
            lbl_path = os.path.join(SRC_LBL_DIR, lbl_file)
            
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(lbl_dist, lbl_file))
                lbl_count += 1

    print(f"Fold {i+1}: {len(fold_keys)} cells | {img_count} images | {lbl_count} labels")

print(f"\nSuccess! All files distributed to {OUTPUT_BASE_DIR}")