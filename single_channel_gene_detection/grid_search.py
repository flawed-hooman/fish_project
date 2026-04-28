import os
import math
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent   # gene_detection_pipeline
BASE_DIR = ROOT.parent                  
BASE_DIR_DATA = BASE_DIR / "FULL_DATASET" / "sky_dataset_5folds"
PRED_ROOT = ROOT / "fusion_results_5fold"

# 4D SEARCH SPACE
MIN_ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
MAX_ALPHAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
K_STEEPNESS = [10, 15, 20]
OFFSETS = [0.55, 0.60, 0.65]

# --- HELPER FUNCTIONS ---
def get_diag(w, h): 
    return math.sqrt(w**2 + h**2)

def get_dist(p1, p2): 
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def read_yolo(path):
    labels = []
    if not path.exists(): return labels
    with open(path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                labels.append({
                    'cid': int(p[0]), 
                    'box': list(map(float, p[1:5])), 
                    'conf': float(p[5]) if len(p) > 5 else 1.0
                })
    return labels

def preload_data():
    all_data = []
    print("Preloading labels into memory...")
    for f_idx in range(1, 6):
        fold_data = []
        gt_dir = BASE_DIR_DATA / f"fold_{f_idx}" / "labels"
        pred_dir = PRED_ROOT / f"fold_{f_idx}" / "predictions" / "labels"
        
        for gt_file in gt_dir.glob("*_SKY_cell*.txt"):
            gt_labs = [l for l in read_yolo(gt_file) if l['cid'] == 3]
            base = gt_file.stem.replace("_SKY_", "_{}_")
            g = read_yolo(pred_dir / (base.format("FITC") + ".txt"))
            r = read_yolo(pred_dir / (base.format("ORANGE") + ".txt"))
            fold_data.append({'gt_count': len(gt_labs), 'greens': g, 'reds': r})
        all_data.append(fold_data)
    return all_data

def run_4d_search_with_fold_details(dataset):
    results = []
    total_combos = len(MIN_ALPHAS) * len(MAX_ALPHAS) * len(K_STEEPNESS) * len(OFFSETS)
    pbar = tqdm(total=total_combos, desc="Grid Searching")

    for min_a in MIN_ALPHAS:
        for max_a in MAX_ALPHAS:
            for k_v in K_STEEPNESS:
                for off in OFFSETS:
                    fold_stats = []
                    
                    for f_idx, fold_data in enumerate(dataset):
                        tp, fp, fn = 0, 0, 0
                        for cell in fold_data:
                            pairs = []
                            # --- 4D SIGMOID FUSION LOGIC ---
                            for i, g in enumerate(cell['greens']):
                                for j, r in enumerate(cell['reds']):
                                    d = get_dist(g['box'][:2], r['box'][:2])
                                    thresh = (get_diag(*g['box'][2:]) + get_diag(*r['box'][2:])) / 2.0
                                    
                                    joint_conf = g['conf'] * r['conf']
                                    # Logistic growth formula
                                    dyn_a = min_a + (max_a - min_a) / (1 + math.exp(-k_v * (joint_conf - off)))
                                    
                                    if d <= (dyn_a * thresh):
                                        pairs.append((d, i, j))
                            
                            pairs.sort()
                            used_g, used_r, n_pred = set(), set(), 0
                            for _, gi, rj in pairs:
                                if gi not in used_g and rj not in used_r:
                                    used_g.add(gi); used_r.add(rj); n_pred += 1
                            
                            # Metrics Accumulation
                            gt = cell['gt_count']
                            tp += min(n_pred, gt)
                            fp += max(0, n_pred - gt)
                            fn += max(0, gt - n_pred)
                        
                        # Fold-level Metrics
                        m_p = tp/(tp+fp) if (tp+fp)>0 else 0
                        m_r = tp/(tp+fn) if (tp+fn)>0 else 0
                        m_f1 = 2*m_p*m_r/(m_p+m_r) if (m_p+m_r)>0 else 0
                        
                        fold_stats.append({
                            'Fold': f_idx + 1,
                            'Mod_P': round(m_p, 4),
                            'Mod_R': round(m_r, 4),
                            'Mod_F1': round(m_f1, 4)
                        })
                    
                    mean_f1 = np.mean([f['Mod_F1'] for f in fold_stats])
                    results.append({
                        'MinA': min_a, 'MaxA': max_a, 'K': k_v, 'Off': off,
                        'Mean_F1': round(mean_f1, 4),
                        'Fold_Details': fold_stats
                    })
                    pbar.update(1)
    
    pbar.close()

    # --- FINAL OUTPUT GENERATION ---
    df_results = pd.DataFrame(results).sort_values('Mean_F1', ascending=False).reset_index(drop=True)
    
    # Save results to CSV
    df_results.to_csv("best_sigmoid_parameters.csv", index=False)
    
    # Extract Best Result 
    best = df_results.iloc[0]
    print("\n" + "="*50)
    print("ABSOLUTE WINNER CONFIGURATION")
    print(f"MinA: {best['MinA']} | MaxA: {best['MaxA']} | K: {best['K']} | Off: {best['Off']}")
    print(f"OVERALL MEAN F1: {best['Mean_F1']}")
    print("="*50)
    print(f"{'Fold':<6} | {'Mod_P':<8} | {'Mod_R':<8} | {'Mod_F1':<8}")
    print("-" * 40)
    for f in best['Fold_Details']:
        print(f"{f['Fold']:<6} | {f['Mod_P']:<8} | {f['Mod_R']:<8} | {f['Mod_F1']:<8}")
    print("-" * 40)
    
    print("\nFull grid search saved to: best_sigmoid_parameters.csv")

if __name__ == "__main__":
    dataset_memory = preload_data()
    if dataset_memory[0]: 
        run_4d_search_with_fold_details(dataset_memory)
    else:
        print("Error: No labels found. Check BASE_DIR and PRED_ROOT paths.")