import os
import sys
from pathlib import Path
import shutil
import yaml

# ================= CONFIGURATION =================
ROOT = Path(__file__).resolve().parent   # gene_detection_pipeline
BASE_DIR = ROOT.parent                  

YOLOV5_DIR = ROOT / "yolov5"
# Trained models 
MODELS_ROOT = ROOT / "gene_results_5fold"
# New inference outputs
OUTPUT_ROOT = ROOT / "fusion_results_5fold"
# Dataset
FOLDS_ROOT = BASE_DIR / "fish_AI_images" / "SKY_ACC_single_probe_images_5folds"

IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = "1"
# =================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
detect_script = YOLOV5_DIR / "detect.py"

# Loop through each fold
for fold_idx in range(1, 6):
    print(f"\n RUNNING CV: FOLD {fold_idx}")
    
    val_path = FOLDS_ROOT / f"fold_{fold_idx}"
    fold_output = MODELS_ROOT / f"fold_{fold_idx}"

    # 2. PREDICT
    best_weights = fold_output / "train" / "weights" / "best.pt"
    if best_weights.exists():
        predict_cmd = (
            f"{sys.executable} {detect_script} "
            f"--weights {best_weights} "
            f"--source {val_path / 'images'} "
            f"--device {DEVICE} "
            f"--img {IMG_SIZE} "
            f"--save-txt --save-conf "
            f"--project {fold_output} --name predictions --exist-ok "
            f"--classes 0 1 2 "
            f"--line-thickness 1 --hide-conf"
        )
        print(f"Executing: {predict_cmd}")
        os.system(predict_cmd)
        


        # 3. ORGANIZE OUTPUTS
        # After detect.py runs, files are at:
        # fold_output/predictions/*.png (The visualisations)
        # fold_output/predictions/labels/*.txt (The label data)
        # DEFINE WHERE YOLO JUST SAVED THE STUFF
        yolo_results_dir = fold_output / "predictions"
        yolo_labels_dir = yolo_results_dir / "labels"

        # 3. DEFINE WHERE THEY GO PERMANENTLY
        final_fold_dir = OUTPUT_ROOT / f"fold_{fold_idx}"
        final_vis_dir = final_fold_dir / "visualisations"
        final_lbl_dir = final_fold_dir / "labels"
        
        final_vis_dir.mkdir(parents=True, exist_ok=True)
        final_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 4. MOVE IMAGES
        for img_file in yolo_results_dir.glob("*.[jp][pn]g"): 
            shutil.move(str(img_file), final_vis_dir / img_file.name)
        
        # 5. MOVE LABELS
        if yolo_labels_dir.exists():
            for lbl_file in yolo_labels_dir.glob("*.txt"):
                shutil.move(str(lbl_file), final_lbl_dir / lbl_file.name)
            
            num_labels = len(list(final_lbl_dir.glob("*.txt")))
            print(f"Fold {fold_idx}: Moved {num_labels} labels to {final_lbl_dir}")
        else:
            print(f"Warning: No labels found in {yolo_labels_dir}")

        print(f"Done. Visuals are in: {final_vis_dir}")