import os
import sys
from pathlib import Path
import shutil
import yaml

# ================= CONFIGURATION =================
ROOT = Path(__file__).resolve().parent   # gene_detection_pipeline
BASE_DIR = ROOT.parent                   # saaransh

YOLOV5_DIR = ROOT / "yolov5"
MODEL = ROOT / "yolov5s.pt"

FOLDS_ROOT = BASE_DIR / "FULL_DATASET" / "gene_dataset_5folds"
OUTPUT_ROOT = ROOT / "gene_results_5fold"

IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = "1"
# =================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
train_script = YOLOV5_DIR / "train.py"
detect_script = YOLOV5_DIR / "detect.py"

def create_fold_yaml(fold_num, train_folders, val_folder):
    """
    Creates a yaml where 'train' is a LIST of 4 folders 
    and 'val' is the single 5th folder.
    """
    content = {
        # YOLOv5 accepts a list of paths for training
        'train': [str(f / "images") for f in train_folders],
        'val': str(val_folder / "images"),
        'nc': 3,  
        'names': ['0', '1', '2'] 
    }
    yaml_path = OUTPUT_ROOT / f"fold_{fold_num}_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f)
    return yaml_path

# ================= 5 FOLD CV =================
for fold_idx in range(1, 6):
    print(f"\n RUNNING CV: FOLD {fold_idx}")
    
    all_fold_paths = [FOLDS_ROOT / f"fold_{i}" for i in range(1, 6)]
    val_path = FOLDS_ROOT / f"fold_{fold_idx}"
    train_paths = [p for p in all_fold_paths if p != val_path]

    current_yaml = create_fold_yaml(fold_idx, train_paths, val_path)
    fold_output = OUTPUT_ROOT / f"fold_{fold_idx}"

    # TRAIN
    train_cmd = (
        f"{sys.executable} {train_script} "
        f"--img {IMG_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} "
        f"--data {current_yaml} --weights {MODEL} " 
        f"--project {fold_output} --name train --device {DEVICE} --exist-ok"
    )
    os.system(train_cmd)

print("\n All 5 folds complete!")