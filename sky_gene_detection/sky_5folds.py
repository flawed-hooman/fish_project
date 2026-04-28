import os
import sys
from pathlib import Path
import shutil
import yaml

# ================= CONFIGURATION =================
ROOT = Path(__file__).resolve().parent   # sky_detection_pipeline
BASE_DIR = ROOT.parent               

YOLOV5_DIR = BASE_DIR / "gene_detection_pipeline" / "yolov5"
FOLDS_ROOT = BASE_DIR / "FULL_DATASET" / "sky_dataset_5folds"
OUTPUT_ROOT = ROOT / "sky_results_5fold"

MODEL = BASE_DIR / "gene_detection_pipeline" / "yolov5s.pt"

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
        'train': [str(f / "images") for f in train_folders],
        'val': str(val_folder / "images"),
        'nc': 4,  
        'names': ['0', '1', '2', '3'] 
    }
    yaml_path = OUTPUT_ROOT / f"fold_{fold_num}_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f)
    return yaml_path

# ================= 5 FOLD CV =================
# Loop through each fold (Dataset has been splitted into 5 folds beforehand)
for fold_idx in range(1, 6):
    print(f"\n RUNNING CV: FOLD {fold_idx}")
    
    all_fold_paths = [FOLDS_ROOT / f"fold_{i}" for i in range(1, 6)]
    val_path = FOLDS_ROOT / f"fold_{fold_idx}"
    train_paths = [p for p in all_fold_paths if p != val_path]

    current_yaml = create_fold_yaml(fold_idx, train_paths, val_path)
    fold_output = OUTPUT_ROOT / f"fold_{fold_idx}"

    # 1. TRAIN
    train_cmd = (
        f"{sys.executable} {train_script} "
        f"--img {IMG_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} "
        f"--data {current_yaml} --weights {MODEL} "
        f"--project {fold_output} --name train --device {DEVICE} --exist-ok"
    )
    os.system(train_cmd)

    # 2. PREDICT
    best_weights = fold_output / "train" / "weights" / "best.pt"
    if best_weights.exists():
        predict_cmd = (
            f"{sys.executable} {detect_script} "
            f"--weights {best_weights} "
            f"--source {val_path / 'images'} "
            f"--img {IMG_SIZE} "
            f"--save-txt "      
            f"--project {fold_output} --name predictions --exist-ok"
        )
        os.system(predict_cmd)
        
        # 3. ORGANISE OUTPUTS
        pred_dir = fold_output / "predictions"
        vis_dir = pred_dir / "visualisations"
        label_out_dir = pred_dir / "labels" 
        
        vis_dir.mkdir(exist_ok=True)
        label_out_dir.mkdir(exist_ok=True)

        # Save Visualisations 
        for img_file in pred_dir.glob("*.jpg"):
            shutil.move(str(img_file), vis_dir / img_file.name)

        print(f"Fold {fold_idx} results: {vis_dir} and {label_out_dir}")
    else:
        print(f"Error: Weights not found for Fold {fold_idx}")

print("\n All 5 folds complete!")
