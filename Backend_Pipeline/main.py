import os
import sys
import zipfile
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Project Imports
from config.settings import CELL_CLASS_ID, FUSION_CLASS_ID, CELL_CONF, GENE_CONF
from fish_utils.image_processing import yolo_to_cv2, cv2_to_yolo, translate_to_global
from fish_utils.fusion import apply_fusion_logic

# Get the absolute path to the yolov5 directory
yolov5_path = Path(__file__).parent / "yolov5"

# Add it to the front of the system path
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))

class IntelliClinixPipeline:
    def __init__(self, nuclei_weights, gene_weights):
        # Loading models from local repository path
        self.nuclei_model = torch.hub.load(str(yolov5_path), 'custom', path=nuclei_weights, source='local')
        self.gene_model = torch.hub.load(str(yolov5_path), 'custom', path=gene_weights, source='local', force_reload=True)
        
        # Setting confidence thresholds
        self.nuclei_model.conf = CELL_CONF
        self.gene_model.conf = GENE_CONF

    def process_zip(self, zip_path, output_root="output_results"):
        temp_dir = Path("temp_extract")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        all_dapi_files = list(temp_dir.rglob("*_DAPI.png"))
        case_dirs = sorted(list(set(f.parent for f in all_dapi_files)))
        
        if not case_dirs:
            print(f"⚠️ No case folders found containing '*_DAPI.png' in {zip_path}.")
            return

        for case_dir in tqdm(case_dirs, desc="Processing Cases"):
            self.process_case(case_dir, Path(output_root) / case_dir.name)

        if temp_dir.exists(): shutil.rmtree(temp_dir)

    def process_case(self, case_dir, output_path):
        # 1. Nuclei Detection
        dapi_img_path = next(case_dir.glob("*_DAPI.png"), None)
        if not dapi_img_path: return
        
        dapi_img = cv2.imread(str(dapi_img_path))
        if dapi_img is None: return
        
        output_path.mkdir(parents=True, exist_ok=True)
        h, w, _ = dapi_img.shape
        
        dapi_img_rgb = cv2.cvtColor(dapi_img, cv2.COLOR_BGR2RGB)
        results = self.nuclei_model(dapi_img_rgb)
        
        global_labels = []
        
        # Color Map for Gene Classes (BGR format)
        gene_class_colors = {
            0: (0, 255, 0),       # Class 0: Green (FITC)
            1: (0, 165, 255),     # Class 1: Orange (ORANGE)
            2: (255, 255, 0),     # Class 2: Cyan (AQUA)
            FUSION_CLASS_ID: (255, 0, 255) # Fusion Class: Magenta (Distinct from Nuclei White)
        }
        
        # Load the SKY channel for visualization
        sky_img_path = next(case_dir.glob("*_SKY.png"), None)
        if sky_img_path:
            sky_vis = cv2.imread(str(sky_img_path))
        else:
            sky_vis = np.zeros((h, w, 3), dtype=np.uint8)

        # 2. Channel Map for Gene detection
        img_map = {
            'FITC': next(case_dir.glob("*_FITC.png"), None),
            'ORANGE': next(case_dir.glob("*_ORANGE.png"), None),
            'AQUA': next(case_dir.glob("*_AQUA.png"), None)
        }

        for idx, det in enumerate(results.xywhn[0]):
            nucleus_yolo = (CELL_CLASS_ID, *det[:4].tolist())
            global_labels.append(nucleus_yolo)
            
            cell_box_pixels = yolo_to_cv2(det[:4].tolist(), (h, w))
            nx1, ny1, nx2, ny2 = cell_box_pixels
            
            # Draw Nuclei: White boundary (Thickness 2)
            cv2.rectangle(sky_vis, (nx1, ny1), (nx2, ny2), (255, 255, 255), 2)

            local_gene_dets = []
            last_patch_shape = None

            for ch_name, ch_path in img_map.items():
                if not ch_path: continue
                
                ch_img = cv2.imread(str(ch_path))
                if ch_img is None: continue
                
                patch = ch_img[max(0, ny1):min(h, ny2), max(0, nx1):min(w, nx2)]
                if patch.size == 0: continue
                
                last_patch_shape = patch.shape
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                gene_results = self.gene_model(patch_rgb)
                
                for g_det in gene_results.xywhn[0]:
                    g_box_px = yolo_to_cv2(g_det[:4].tolist(), patch.shape)
                    local_gene_dets.append({'cid': int(g_det[5]), 'box': g_box_px})

            # 3. Fusion & Translation
            if local_gene_dets and last_patch_shape:
                fused_dets = apply_fusion_logic(local_gene_dets, last_patch_shape)
                for f_det in fused_dets:
                    global_px = translate_to_global(f_det['box'], cell_box_pixels)
                    global_yolo = cv2_to_yolo(global_px, (h, w), f_det['cid'])
                    global_labels.append(global_yolo)
                    
                    # Draw Fused Result: Class-specific color (Thickness 1)
                    gx1, gy1, gx2, gy2 = global_px
                    color = gene_class_colors.get(f_det['cid'], (0, 0, 255))
                    cv2.rectangle(sky_vis, (gx1, gy1), (gx2, gy2), color, 1)

        # Save Final SKY Visualization
        vis_save_name = dapi_img_path.name.replace("_DAPI.png", "_SKY_VIS.png")
        cv2.imwrite(str(output_path / vis_save_name), sky_vis)

        # 4. Global SKY Label Save
        sky_label_name = dapi_img_path.name.replace("_DAPI.png", "_SKY.txt")
        sky_label_path = output_path / sky_label_name
        with open(sky_label_path, 'w') as f:
            for lbl in global_labels:
                f.write(f"{int(lbl[0])} {' '.join(f'{x:.6f}' for x in lbl[1:])}\n")

if __name__ == "__main__":
    pipeline = IntelliClinixPipeline(
        nuclei_weights="models/nuclei_detection.pt",
        gene_weights="models/gene_detection.pt"
    )
    pipeline.process_zip("input_data.zip")
    print("\n✅ Multi-stage detection complete: Final SKY visualizations (White Nuclei / Magenta Fusions) saved.")