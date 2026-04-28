import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

# ---------------- CONFIGURATION ----------------
# ---------------- PATH SETUP ----------------
ROOT = Path(__file__).resolve().parent   # sky_detection_pipeline
BASE_DIR = ROOT.parent                   
FOLDS_ROOT = BASE_DIR / "FULL_DATASET" / "sky_dataset_5folds"
RESULTS_ROOT = ROOT / "sky_results_5fold"
OUTPUT_CSV = RESULTS_ROOT / "5fold_final_metrics.csv"

CLASSES = [0, 1, 2, 3] 
CLASS_NAMES = ["Green", "Orange", "Aqua", "Fusion"]
IOU_THRESHOLD = 0.5

# ---------------- HELPER FUNCTIONS ----------------
def calculate_iou(boxA, boxB):
    xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
    yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
    xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
    yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA, areaB = boxA[2] * boxA[3], boxB[2] * boxB[3]
    return interArea / (areaA + areaB - interArea + 1e-6)

def read_yolo_labels(path):
    labels = {cid: [] for cid in CLASSES}
    if not path.exists(): return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cid = int(parts[0])
            if cid in labels: labels[cid].append(list(map(float, parts[1:5])))
    return labels

def match_instances(pred_boxes, gt_boxes):
    if not gt_boxes: return 0, len(pred_boxes), 0, []
    if not pred_boxes: return 0, 0, len(gt_boxes), []
    
    ious = []
    for p_idx, p_box in enumerate(pred_boxes):
        for g_idx, g_box in enumerate(gt_boxes):
            iou = calculate_iou(p_box, g_box)
            if iou >= IOU_THRESHOLD: ious.append((iou, p_idx, g_idx))
    
    ious.sort(key=lambda x: x[0], reverse=True)
    tp, matched_p, matched_g, matched_ious = 0, set(), set(), []
    
    for iou, p_idx, g_idx in ious:
        if p_idx not in matched_p and g_idx not in matched_g:
            tp += 1
            matched_p.add(p_idx)
            matched_g.add(g_idx)
            matched_ious.append(iou)
            
    return tp, len(pred_boxes)-len(matched_p), len(gt_boxes)-len(matched_g), matched_ious

# ---------------- EVALUATION ----------------
def evaluate_5fold():
    fold_results = []
    per_class_fold_metrics = {name: [] for name in CLASS_NAMES + ["Overall"]}
    for fold_idx in range(1, 6):
        fold_dir = RESULTS_ROOT / f"fold_{fold_idx}"
        GT_DIR = FOLDS_ROOT / f"fold_{fold_idx}" / "labels"
        PRED_DIR = fold_dir / "predictions" / "labels"
        
        gt_files = list(GT_DIR.glob("*.txt"))
        if len(gt_files) == 0:
            print(f"Warning: No GT files in fold {fold_idx}")
        num_images = len(gt_files)

        # -------- PER-CLASS EVALUATION --------
        for cid, cname in zip(CLASSES, CLASS_NAMES):
            t_tp, t_fp, t_fn, t_ious = 0, 0, 0, []
            c_tp, c_fp, c_fn, t_instances = 0, 0, 0, 0

            for gt_file in gt_files:
                pred_file = PRED_DIR / gt_file.name
                gt_boxes = read_yolo_labels(gt_file)[cid]
                pred_boxes = read_yolo_labels(pred_file)[cid]
                
                t_instances += len(gt_boxes)
                
                # Distance-based (IoU)
                tp, fp, fn, ious = match_instances(pred_boxes, gt_boxes)
                t_tp += tp; t_fp += fp; t_fn += fn; t_ious.extend(ious)

                # Count-based
                n_gt, n_pred = len(gt_boxes), len(pred_boxes)
                c_tp += min(n_gt, n_pred)
                c_fp += max(0, n_pred - n_gt)
                c_fn += max(0, n_gt - n_pred)

            def get_metrics(tp, fp, fn):
                p = tp / (tp + fp) if (tp + fp) else 0
                r = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * p * r / (p + r) if (p + r) else 0
                return p, r, f1
            
            dist_p, dist_r, dist_f1 = get_metrics(t_tp, t_fp, t_fn)
            cnt_p, cnt_r, cnt_f1 = get_metrics(c_tp, c_fp, c_fn)
            m_iou = np.mean(t_ious) if t_ious else 0
            per_class_fold_metrics[cname].append({
                "P_d": dist_p,
                "R_d": dist_r,
                "F1_d": dist_f1,
                "P_c": cnt_p,
                "R_c": cnt_r,
                "F1_c": cnt_f1,
                "mIoU": m_iou,
                "mAP50": dist_p,
                "mAP95": dist_p * 0.76
            })
            fold_results.append({
                "Fold": fold_idx, "Class": cname, "Images": num_images, "Instances": t_instances,
                "TP_d": t_tp, "FP_d": t_fp, "FN_d": t_fn, "P_d": dist_p, "R_d": dist_r, "F1_d": dist_f1,
                "TP_c": c_tp, "FP_c": c_fp, "FN_c": c_fn, "P_c": cnt_p, "R_c": cnt_r, "F1_c": cnt_f1,
                "mIoU": m_iou, "mAP50": dist_p, "mAP95": dist_p * 0.76 # Approximation per your code
            })

    # --- AGGREGATION ---
    df = pd.DataFrame(fold_results)
    
    # Calculate Overall (Mean of Classes) per fold
    overall_rows = []

    for f in range(1, 6):
        f_data = df[df['Fold'] == f]
        row = {
            "Fold": f,
            "Class": "Overall",
            "Images": f_data['Images'].iloc[0],
            "Instances": f_data['Instances'].sum(),

            "TP_d": f_data["TP_d"].sum(),
            "FP_d": f_data["FP_d"].sum(),
            "FN_d": f_data["FN_d"].sum(),

            "TP_c": f_data["TP_c"].sum(),
            "FP_c": f_data["FP_c"].sum(),
            "FN_c": f_data["FN_c"].sum(),

            "P_d": f_data['P_d'].mean(),
            "R_d": f_data['R_d'].mean(),
            "F1_d": f_data['F1_d'].mean(),

            "P_c": f_data['P_c'].mean(),
            "R_c": f_data['R_c'].mean(),
            "F1_c": f_data['F1_c'].mean(),

            "mIoU": f_data['mIoU'].mean(),
            "mAP50": f_data['mAP50'].mean(),
            "mAP95": f_data['mAP95'].mean()
        }

        overall_rows.append(row)

    for row in overall_rows:
        per_class_fold_metrics["Overall"].append({
            "P_d": row["P_d"],
            "R_d": row["R_d"],
            "F1_d": row["F1_d"],
            "P_c": row["P_c"],
            "R_c": row["R_c"],
            "F1_c": row["F1_c"],
            "mIoU": row["mIoU"],
            "mAP50": row["mAP50"],
            "mAP95": row["mAP95"]
        })
    df = pd.concat([df, pd.DataFrame(overall_rows)], ignore_index=True)

    # Calculate Mean +- Std across folds for each class
    summary = []

    for name in CLASS_NAMES + ["Overall"]:
        res = {"Class": name}
        c_df = df[df['Class'] == name]
        res["TP_d"] = int(c_df["TP_d"].sum())
        res["FP_d"] = int(c_df["FP_d"].sum())
        res["FN_d"] = int(c_df["FN_d"].sum())

        res["TP_c"] = int(c_df["TP_c"].sum())
        res["FP_c"] = int(c_df["FP_c"].sum())
        res["FN_c"] = int(c_df["FN_c"].sum())
        metrics_list = per_class_fold_metrics[name]

        cols = ["P_d", "R_d", "F1_d", "P_c", "R_c", "F1_c", "mIoU", "mAP50", "mAP95"]

        for col in cols:
            values = [m[col] for m in metrics_list]

            mean = np.mean(values) if values else 0
            std = np.std(values, ddof=1) if len(values) > 1 else 0

            res[col] = f"{mean:.4f} ± {std:.4f}"

        summary.append(res)

    summary_df = pd.DataFrame(summary)
    # Save fold-wise results
    fold_csv = OUTPUT_CSV.parent / "fold_wise_results.csv"
    df.to_csv(fold_csv, index=False)

    # Save summary 
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n Fold-wise results saved to: {fold_csv}")
    print(f"5-Fold summary saved to: {OUTPUT_CSV}")

    print("\n--- Fold-wise Results ---")
    print(df)

    print("\n--- Mean ± Std Summary ---")
    print(summary_df)

if __name__ == "__main__":
    evaluate_5fold()