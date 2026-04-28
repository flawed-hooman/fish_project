import os
import cv2
import math
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent   # gene_detection_pipeline
BASE_DIR = ROOT.parent                  
BASE_DIR_DATA = BASE_DIR / "FULL_DATASET" / "sky_dataset_5folds"
OUTPUT_ROOT = ROOT / "fusion_results_5fold"

BEST_S_MIN = 1.0  # min_a
BEST_S_MAX = 1.6  # max_a
K_V = 10        # logistic growth
OFF = 0.55

CLASS_MAP = {'FITC': 0, 'ORANGE': 1, 'AQUA': 2, 'FUSION': 3}
COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] #Fusion(Yellow)

# ------ HELPER FUNCTIONS  -------
def get_center(yolo_box):
    return (yolo_box[0], yolo_box[1])

def get_diag(yolo_box):
    return math.sqrt(yolo_box[2]**2 + yolo_box[3]**2)

def get_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def read_yolo_labels(path):
    labels = []
    if not path.exists(): return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            labels.append({
                'cid': int(parts[0]),
                'yolo': list(map(float, parts[1:5])),
                'conf': float(parts[5]) if len(parts) > 5 else 1.0
            })
    return labels

def draw_boxes(img, elements):
    h, w, _ = img.shape
    for cid, box in elements:
        cx, cy, bw, bh = box
        # Convert YOLO normalized to Pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # 1. Draw only the bounding box 
        color = COLORS[cid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1) 
        
        # 2. Put the ID number (0, 1, 2, or 3) just above the box
        label = str(cid)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        
        # Position the number slightly above the top-left corner
        cv2.putText(img, label, (x1, y1 - 3), font, scale, color, thickness, cv2.LINE_AA)
        
    return img

def save_full_sky_labels_and_vis():
    for fold_idx in range(1, 6):
        fold_path = BASE_DIR_DATA / f"fold_{fold_idx}"
        pred_root = OUTPUT_ROOT / f"fold_{fold_idx}" / "predictions" / "labels"
        final_label_dir = OUTPUT_ROOT / f"fold_{fold_idx}" / "sky_labels"
        vis_dir = OUTPUT_ROOT / f"fold_{fold_idx}" / "visualizations"
        
        final_label_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        img_files = list((fold_path / "images").glob("*_SKY_cell*"))
        
        for img_path in tqdm(img_files, desc=f"Processing Fold {fold_idx}"):
            base_name = img_path.stem
            
            # 1. Load Data
            greens = read_yolo_labels(pred_root / (base_name.replace("_SKY_", "_FITC_") + ".txt"))
            reds = read_yolo_labels(pred_root / (base_name.replace("_SKY_", "_ORANGE_") + ".txt"))
            aquas = read_yolo_labels(pred_root / (base_name.replace("_SKY_", "_AQUA_") + ".txt"))

            # 2. Fusion Logic (UPDATED WITH LOGISTIC GROWTH)
            pairs = []
            for i, g in enumerate(greens):
                for j, r in enumerate(reds):
                    # Distance between centers
                    dist = get_distance(get_center(g['yolo']), get_center(r['yolo']))
                    
                    # Threshold based on diagonals
                    thresh = (get_diag(g['yolo']) + get_diag(r['yolo'])) / 2.0
                    
                    # Joint Confidence
                    joint_conf = g['conf'] * r['conf']
                    
                    # Logistic growth formula for dynamic scaling factor (dyn_a)
                    dyn_a = BEST_S_MIN + (BEST_S_MAX - BEST_S_MIN) / (1 + math.exp(-K_V * (joint_conf - OFF)))
                    
                    # Check if within dynamic distance threshold
                    if dist <= (dyn_a * thresh):
                        pairs.append((dist, i, j))
            
            pairs.sort()
            used_g, used_r, final_elements = set(), set(), []

            # pairs = []
            # for i, g in enumerate(greens):
            #     for j, r in enumerate(reds):
            #         dist = get_distance(get_center(g['yolo']), get_center(r['yolo']))
            #         base_thresh = (get_diag(g['yolo']) + get_diag(r['yolo'])) / 2.0
            #         joint_conf = g['conf'] * r['conf']
            #         norm_c = max(0.0, (joint_conf - 0.0625) / (1.0 - 0.0625))
            #         scale_factor = BEST_S_MIN + (norm_c * (BEST_S_MAX - BEST_S_MIN))
                    
            #         if dist <= (base_thresh * scale_factor):
            #             pairs.append((dist, i, j))
            
            # pairs.sort()
            # used_g, used_r, final_elements = set(), set(), []

            for dist, gi, rj in pairs:
                if gi in used_g or rj in used_r: continue
                used_g.add(gi); used_r.add(rj)
                g_box, r_box = greens[gi]['yolo'], reds[rj]['yolo']
                fusion_box = [(g_box[k] + r_box[k]) / 2.0 for k in range(4)]
                final_elements.append((CLASS_MAP['FUSION'], fusion_box))

            for i, g in enumerate(greens):
                if i not in used_g: final_elements.append((CLASS_MAP['FITC'], g['yolo']))
            for i, r in enumerate(reds):
                if i not in used_r: final_elements.append((CLASS_MAP['ORANGE'], r['yolo']))
            for a in aquas: 
                final_elements.append((CLASS_MAP['AQUA'], a['yolo']))

            # 3. Save Labels
            with open(final_label_dir / (base_name + ".txt"), "w") as f:
                for cid, box in final_elements:
                    f.write(f"{cid} {' '.join(map(str, box))}\n")

            # 4. Visualizations
            img = cv2.imread(str(img_path))
            if img is not None:
                img_vis = draw_boxes(img, final_elements)
                cv2.imwrite(str(vis_dir / (base_name + ".png")), img_vis)

    print(f"✅ Process Complete. Labels in 'sky_labels' and images in 'visualizations'.")

if __name__ == "__main__":
    save_full_sky_labels_and_vis()