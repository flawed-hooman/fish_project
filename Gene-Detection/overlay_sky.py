import cv2
import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("FISH-Sample-Standardized") 
CHANNELS = ['FITC', 'ORANGE', 'AQUA']
FUSION_CLASS_ID = 3
NUM_FOLDS = 5

# Define your Grid Search Hyperparameters here
S_MIN_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]  # Values for the weakest signals
S_MAX_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # Values for the strongest signals

# --- HELPER FUNCTIONS ---
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def get_distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def yolo_to_cv2(yolo_box, img_shape):
    h, w = img_shape
    cx_n, cy_n, w_n, h_n = yolo_box
    x1 = int((cx_n - w_n/2) * w)
    y1 = int((cy_n - h_n/2) * h)
    x2 = int((cx_n + w_n/2) * w)
    y2 = int((cy_n + h_n/2) * h)
    return (x1, y1, x2, y2)

def read_yolo_labels(path, target_class=None):
    labels = []
    if not path.exists():
        return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if target_class is None or cls_id == target_class:
                box = list(map(float, parts[1:5]))
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                labels.append({'cid': cls_id, 'yolo': box, 'conf': conf})
    return labels

def preload_data():
    """Reads all necessary data into memory once to make the grid search instant."""
    print("Pre-loading data into memory...")
    dataset = []
    
    case_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name != "yolo_ground_truth"])

    for case_dir in tqdm(case_dirs, desc="Loading Cases"):
        gt_dir = case_dir / "yolo_ground_truth"
        patch_dir = case_dir / "cell_patches"
        pred_dir = case_dir / "predictions" / "labels"
        
        if not gt_dir.exists() or not patch_dir.exists() or not pred_dir.exists(): 
            continue

        sky_gt_files = list(gt_dir.glob("*_SKY_cell*.txt"))
        
        for gt_file in sky_gt_files:
            # Get Ground Truth Fusions
            gt_fusions = read_yolo_labels(gt_file, FUSION_CLASS_ID)
            num_gt = len(gt_fusions)
            
            # Reconstruct names to find the SKY image and individual channel predictions
            base_name = gt_file.name.replace("_SKY_", "_{}_")
            sky_img_name = gt_file.with_suffix(".png").name
            sky_img_path = patch_dir / sky_img_name
            
            if not sky_img_path.exists(): continue
            
            # Read image shape once
            img_bgr = cv2.imread(str(sky_img_path))
            if img_bgr is None: continue
            img_h, img_w, _ = img_bgr.shape
            
            # Load Predictions for FITC and ORANGE
            greens, reds = [], []
            
            fitc_txt = pred_dir / base_name.format("FITC")
            orange_txt = pred_dir / base_name.format("ORANGE")
            
            if fitc_txt.exists():
                for lab in read_yolo_labels(fitc_txt, 0): # 0 is FITC
                    cv2_box = yolo_to_cv2(lab['yolo'], (img_h, img_w))
                    greens.append({'box': cv2_box, 'conf': lab['conf']})
                    
            if orange_txt.exists():
                for lab in read_yolo_labels(orange_txt, 1): # 1 is ORANGE
                    cv2_box = yolo_to_cv2(lab['yolo'], (img_h, img_w))
                    reds.append({'box': cv2_box, 'conf': lab['conf']})
            
            dataset.append({
                'cell_name': sky_img_name,
                'num_gt': num_gt,
                'greens': greens,
                'reds': reds
            })
            
    return dataset

def run_grid_search(dataset):
    print(f"\nRunning Grid Search over {len(S_MIN_GRID) * len(S_MAX_GRID)} combinations with {NUM_FOLDS}-Fold CV...")
    results = []
    
    # Split dataset into folds for mean/standard deviation calculation
    folds = np.array_split(dataset, NUM_FOLDS)
    
    for s_min in S_MIN_GRID:
        for s_max in S_MAX_GRID:
            # Skip invalid configurations where min is greater than max
            if s_min > s_max: 
                continue
                
            fold_metrics = []
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for fold_data in folds:
                mod_tp, mod_fp, mod_fn = 0, 0, 0
                
                for data in fold_data:
                    greens = data['greens']
                    reds = data['reds']
                    num_gt = data['num_gt']
                    
                    # --- Fusion Logic for this Hyperparameter Pair ---
                    pairs = []
                    for i, g in enumerate(greens):
                        for j, r in enumerate(reds):
                            dist = get_distance(get_center(g['box']), get_center(r['box']))
                            base_thresh = (get_diag(g['box']) + get_diag(r['box'])) / 2.0
                            
                            joint_conf = g['conf'] * r['conf']
                            norm_c = max(0.0, (joint_conf - 0.0625) / (1.0 - 0.0625))
                            
                            scale_factor = s_min + (norm_c * (s_max - s_min))
                            dynamic_thresh = base_thresh * scale_factor
                            
                            if dist <= dynamic_thresh:
                                pairs.append((dist, i, j))
                        
                    pairs.sort()
                    used_g, used_r = set(), set()
                    num_pred = 0
                    
                    for dist, gi, rj in pairs:
                        if gi in used_g or rj in used_r: continue
                        used_g.add(gi)
                        used_r.add(rj)
                        num_pred += 1
                        
                    # --- Metrics Logic ---
                    if num_pred == num_gt:
                        mod_tp += num_gt
                    elif num_pred > num_gt:
                        mod_tp += num_gt           
                        mod_fp += (num_pred - num_gt) 
                    elif num_pred < num_gt:
                        mod_tp += num_pred         
                        mod_fn += (num_gt - num_pred)

                # Store fold-level metrics
                fold_p = mod_tp / (mod_tp + mod_fp) if (mod_tp + mod_fp) > 0 else 0
                fold_r = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0
                fold_f1 = 2 * (fold_p * fold_r) / (fold_p + fold_r) if (fold_p + fold_r) > 0 else 0
                fold_metrics.append({'p': fold_p, 'r': fold_r, 'f1': fold_f1})
                
                # Accumulate for raw global counts
                total_tp += mod_tp
                total_fp += mod_fp
                total_fn += mod_fn

            # --- Calculate Mean Metrics across folds ---
            mean_p = np.mean([fm['p'] for fm in fold_metrics]) if fold_metrics else 0
            mean_r = np.mean([fm['r'] for fm in fold_metrics]) if fold_metrics else 0
            mean_f1 = np.mean([fm['f1'] for fm in fold_metrics]) if fold_metrics else 0
            
            # --- Calculate Standard Deviations across folds ---
            std_p = np.std([fm['p'] for fm in fold_metrics], ddof=1) if len(fold_metrics) > 1 else 0
            std_r = np.std([fm['r'] for fm in fold_metrics], ddof=1) if len(fold_metrics) > 1 else 0
            std_f1 = np.std([fm['f1'] for fm in fold_metrics], ddof=1) if len(fold_metrics) > 1 else 0
            
            results.append({
                "S_Min": s_min,
                "S_Max": s_max,
                "Total_TP": total_tp,
                "Total_FP": total_fp,
                "Total_FN": total_fn,
                "Mean_Precision": round(mean_p, 4),
                "Std_Precision": round(std_p, 4),
                "Mean_Recall": round(mean_r, 4),
                "Std_Recall": round(std_r, 4),
                "Mean_F1_Score": round(mean_f1, 4),
                "Std_F1_Score": round(std_f1, 4)
            })
            
    # Compile Results
    df = pd.DataFrame(results)
    # Sort by the best Mean F1 score descending
    df = df.sort_values(by="Mean_F1_Score", ascending=False).reset_index(drop=True)
    
    # Print the Top 5 Configurations
    print("\n--- TOP 5 CONFIGURATIONS (Sorted by Mean F1 Score) ---")
    print(df.head(5).to_string(index=False))
    
    df.to_csv("fusion_grid_search_metrics_5fold_macro.csv", index=False)
    print("\n✅ Grid Search complete. All results saved to 'fusion_grid_search_metrics_5fold_macro.csv'.")

if __name__ == "__main__":
    dataset_memory = preload_data()
    if dataset_memory:
        run_grid_search(dataset_memory)
    else:
        print("❌ No data loaded. Check directory paths.")