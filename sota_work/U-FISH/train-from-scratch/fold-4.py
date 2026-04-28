#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  FOLD-4 TRAINING SCRIPT — U-FISH From Scratch (Random Kaiming-He Init)
#
#  5-Fold Cross-Validation:
#    Validation : fold-4
#    Training   : fold-1 + fold-2 + fold-3 + fold-5
#
#  Architecture : UFishNet (Nature Methods 2024)
#  Loss         : DiceRMSELoss = 0.6 × DiceLoss + 0.4 × RMSELoss
#  Optimizer    : Adam (lr=1e-3, NO weight decay, NO LR scheduler)
#  Weights      : Random Kaiming-He initialisation (from scratch)
#
#  Output directory: results-4/
#    - epoch_001.pth ... epoch_050.pth   (model checkpoint every epoch)
#    - best_model.pth                    (copy of the best epoch's weights)
#    - metrics_per_epoch.csv             (detailed per-epoch metrics)
#    - training_config.json              (full config for reproducibility)
#    - training_summary.json             (complete results)
#
#  For MICCAI submission — follows train.ipynb procedure exactly.
# ══════════════════════════════════════════════════════════════════════════════

import copy
import csv
import glob
import json
import math
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import maximum_filter, label as ndimage_label
from torch.utils.data import DataLoader, Dataset
from ufish.api import UFish

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
VAL_FOLD    = 4                  # THIS fold is used for validation
N_FOLDS     = 5
FOLD_ROOT   = "/home/sukrit/saaransh/fish_project_sota_work/Model_B/Gene-Dataset-5Fold-Merged"
FOLD_FMT    = "fold-{k}"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, f"results-{VAL_FOLD}")

EPOCHS      = 50
BATCH_SIZE  = 4
IMG_SIZE    = 512
NUM_WORKERS = 0
LR          = 1e-3              # from-scratch learning rate (higher than finetuned)

INTENSITY_THRESHOLD = 0.5       # for peak detection in evaluation

CLASS_NAMES  = {0: "green", 1: "red", 2: "aqua"}
CLASS_COLORS = {0: "#39FF14", 1: "#FF6B6B", 2: "#00E5FF"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE UTILITY FUNCTIONS (exact copies from train.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def parse_yolo_to_spots(label_path: str, img_w: int, img_h: int) -> list:
    """
    Read a YOLO-format label file → list of spot dicts in pixel coordinates.

    YOLO format: [class_id  cx  cy  w  h]  (all normalised 0-1)
    σ = FWHM / 2.3548,  FWHM = sqrt(w_px × h_px)
    """
    spots = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls   = int(parts[0])
            cx    = float(parts[1]) * img_w
            cy    = float(parts[2]) * img_h
            w_px  = float(parts[3]) * img_w
            h_px  = float(parts[4]) * img_h
            fwhm  = np.sqrt(w_px * h_px)
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            spots.append(dict(cls=cls, cx_px=cx, cy_px=cy,
                              w_px=w_px, h_px=h_px,
                              fwhm_px=fwhm, sigma_px=sigma))
    return spots


def make_density_map(spots: list, img_h: int, img_w: int,
                     fixed_sigma_by_class: dict = None,
                     fixed_sigma: float = None) -> np.ndarray:
    """
    Build a Gaussian density map — the U-FISH training TARGET.

    Each spot → 2D Gaussian (peak=1.0). Overlapping Gaussians are max-pooled.
    σ selection: per-class fixed > global fixed > per-bbox.
    ±4σ window for efficiency.
    """
    density = np.zeros((img_h, img_w), dtype=np.float32)
    if len(spots) == 0:
        return density

    rr, cc = np.mgrid[0:img_h, 0:img_w].astype(np.float32)

    for s in spots:
        cx, cy = s["cx_px"], s["cy_px"]

        if fixed_sigma_by_class and s["cls"] in fixed_sigma_by_class:
            sigma = fixed_sigma_by_class[s["cls"]]
        elif fixed_sigma is not None:
            sigma = fixed_sigma
        else:
            sigma = s["sigma_px"]

        r0 = max(0, int(cy - 4 * sigma));  r1 = min(img_h, int(cy + 4 * sigma) + 1)
        c0 = max(0, int(cx - 4 * sigma));  c1 = min(img_w, int(cx + 4 * sigma) + 1)
        if r0 >= r1 or c0 >= c1:
            continue

        rr_sub = rr[r0:r1, c0:c1]
        cc_sub = cc[r0:r1, c0:c1]
        gauss  = np.exp(-((cc_sub - cx)**2 + (rr_sub - cy)**2) / (2.0 * sigma**2))
        density[r0:r1, c0:c1] = np.maximum(density[r0:r1, c0:c1], gauss)

    return density


def get_probe_channel(filename: str) -> int:
    """Detect which RGB channel holds the probe signal from the filename."""
    name = os.path.basename(filename).upper()
    if "ORANGE" in name:
        return 0          # Red channel (class 1)
    elif "FITC" in name or "GREEN" in name:
        return 1          # Green channel (class 0)
    elif "AQUA" in name:
        return 2          # Blue channel (class 2)
    return 0              # default to Red


def load_and_preprocess(img_path: str):
    """
    Load image → extract probe channel → DAPI-subtract → float32 [0,1].

    DAPI (blue channel) subtracted from non-blue probes.
    Aqua (class 2, blue) skips DAPI subtraction.
    Returns: (probe_sub, img_rgb, probe_ch)
    """
    img_rgb  = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    img_f    = img_rgb.astype(np.float32) / 255.0
    probe_ch = get_probe_channel(img_path)
    probe    = img_f[:, :, probe_ch]
    dapi     = img_f[:, :, 2]

    if probe_ch == 2:
        probe_sub = probe
    else:
        probe_sub = np.clip(probe - dapi, 0, 1)

    return probe_sub, img_rgb, probe_ch


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION — DiceRMSELoss (EXACTLY as in U-FISH paper, train.ipynb)
#
#  L = 0.6 × DiceLoss + 0.4 × RMSELoss
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft Dice loss for continuous density maps (U-FISH)."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum()
        union = (pred ** 2).sum() + (target ** 2).sum()
        return 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss (U-FISH)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(nn.functional.mse_loss(pred, target))


class DiceRMSELoss(nn.Module):
    """
    U-FISH paper loss: 0.6 × DiceLoss + 0.4 × RMSELoss.
    Exactly matches ufish.model.loss.DiceRMSELoss from the official package.
    """
    def __init__(self, dice_weight: float = 0.6, rmse_weight: float = 0.4,
                 eps: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.rmse_weight = rmse_weight
        self.dice_loss = DiceLoss(eps=eps)
        self.rmse_loss = RMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice_loss(pred, target) +
                self.rmse_weight * self.rmse_loss(pred, target))


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET CLASS (exact copy from train.ipynb Cell 3)
#
#  Input  = DAPI-subtracted probe channel  (1, 512, 512) float32 [0,1]
#  Target = Gaussian density map           (1, 512, 512) float32 [0,1]
#  Augmentation: H/V flip only (no crop — avoids empty patches)
# ══════════════════════════════════════════════════════════════════════════════

class FISHDataset(Dataset):
    def __init__(self, samples: list,
                 fixed_sigma_by_class: dict = None,
                 fixed_sigma: float = None,
                 augment: bool = True):
        self.samples              = samples
        self.fixed_sigma_by_class = fixed_sigma_by_class
        self.fixed_sigma          = fixed_sigma
        self.augment              = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        probe_sub, _, _ = load_and_preprocess(s["img_path"])
        H, W = probe_sub.shape

        spots   = parse_yolo_to_spots(s["label_path"], W, H)
        density = make_density_map(spots, H, W,
                                   fixed_sigma_by_class=self.fixed_sigma_by_class,
                                   fixed_sigma=self.fixed_sigma)

        if self.augment:
            if np.random.random() > 0.5:
                probe_sub = np.flip(probe_sub, axis=1).copy()
                density   = np.flip(density,   axis=1).copy()
            if np.random.random() > 0.5:
                probe_sub = np.flip(probe_sub, axis=0).copy()
                density   = np.flip(density,   axis=0).copy()

        inp = torch.from_numpy(probe_sub[np.newaxis]).float()   # (1, H, W)
        tgt = torch.from_numpy(density[np.newaxis]).float()     # (1, H, W)
        return inp, tgt


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET DISCOVERY (exact copy from train.ipynb Cell 4)
# ══════════════════════════════════════════════════════════════════════════════

def discover_samples(root_dir: str, fold_name: str) -> list:
    """Scan one fold directory → list of sample dicts."""
    img_dir   = os.path.join(root_dir, fold_name, "images")
    label_dir = os.path.join(root_dir, fold_name, "labels")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.png")))
    samples, missing = [], 0
    for img_path in img_paths:
        stem       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(label_path):
            missing += 1
            continue
        probe_ch = get_probe_channel(img_path)
        cls_id, n = -1, 0
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    n += 1
        samples.append(dict(img_path=img_path, label_path=label_path,
                            probe_ch=probe_ch, n_spots=n, cls=cls_id))
    if missing:
        print(f"    Warning: {missing} images skipped (no matching label)")
    return samples


def compute_fixed_sigma_by_class(fold_samples: dict, n_folds: int) -> dict:
    """Compute per-class fixed σ = median(σ) from ALL fold annotations (Cell 5)."""
    all_sigmas = []
    sigma_by_cls = {0: [], 1: [], 2: []}

    for k in range(1, n_folds + 1):
        for s in fold_samples[k]:
            img_w, img_h = Image.open(s["img_path"]).size
            spots = parse_yolo_to_spots(s["label_path"], img_w, img_h)
            for sp in spots:
                all_sigmas.append(sp["sigma_px"])
                if sp["cls"] in sigma_by_cls:
                    sigma_by_cls[sp["cls"]].append(sp["sigma_px"])

    if len(all_sigmas) == 0:
        return {0: 2.0, 1: 2.0, 2: 2.0}

    all_sigmas = np.array(all_sigmas)
    fixed = {}
    for cls_id in [0, 1, 2]:
        arr = np.array(sigma_by_cls[cls_id])
        if len(arr) > 0:
            fixed[cls_id] = float(np.median(arr))
        else:
            fixed[cls_id] = float(np.median(all_sigmas))
    return fixed


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS (from train.ipynb Cell 9)
#
#  Gaussian-IoU Proxy for Spot Detection:
#    IoU_proxy = max(0, 1 − dist / detection_radius)
#    detection_radius = FWHM / 2 = √(w_gt × h_gt) / 2
# ══════════════════════════════════════════════════════════════════════════════

def detect_spots_from_density(density_map: np.ndarray,
                              threshold: float = 0.5,
                              min_distance: int = 3) -> list:
    """
    Detect spots from a predicted density map using local maxima detection.

    Returns list of (x, y, confidence) tuples.
    """
    if density_map.ndim == 3:
        density_map = density_map.squeeze()

    # Normalise to [0, 1]
    d_min, d_max = density_map.min(), density_map.max()
    if d_max - d_min > 1e-8:
        density_norm = (density_map - d_min) / (d_max - d_min)
    else:
        density_norm = density_map

    # Local maxima detection via maximum filter
    local_max = maximum_filter(density_norm, size=2 * min_distance + 1)
    detected = (density_norm == local_max) & (density_norm >= threshold)

    coords = np.argwhere(detected)  # (row, col) = (y, x)
    spots = []
    for (y, x) in coords:
        conf = float(density_norm[y, x])
        spots.append((float(x), float(y), conf))

    return spots


def gaussian_iou_proxy(pred_xy, gt_xy, gt_w, gt_h):
    """
    IoU-equivalent for Gaussian blob detection.
    detection_radius = FWHM / 2 = sqrt(w_gt * h_gt) / 2
    IoU_proxy = max(0, 1 - dist / detection_radius)
    """
    dist = np.sqrt((pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2)
    radius = np.sqrt(gt_w * gt_h) / 2.0
    return max(0.0, 1.0 - dist / max(radius, 1e-8))


def compute_ap_at_iou(predictions, gt_dict, n_gt, iou_thresh):
    """
    Compute Average Precision at a given IoU threshold.

    predictions : list of (confidence, pred_x, pred_y, image_idx) sorted desc
    gt_dict     : dict image_idx → [(cx, cy, w, h), ...]
    n_gt        : int total GT count
    iou_thresh  : float

    Returns: (AP, final_precision, final_recall)
    """
    if n_gt == 0 or len(predictions) == 0:
        return 0.0, 0.0, 0.0

    matched = {idx: [False] * len(gts) for idx, gts in gt_dict.items()}
    tp_list, fp_list = [], []

    for conf, px, py, img_idx in predictions:
        gts = gt_dict.get(img_idx, [])
        best_iou, best_j = -1, -1

        for j, (cx, cy, w, h) in enumerate(gts):
            iou = gaussian_iou_proxy((px, py), (cx, cy), w, h)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thresh and best_j >= 0 and not matched[img_idx][best_j]:
            tp_list.append(1)
            fp_list.append(0)
            matched[img_idx][best_j] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list).astype(float)
    fp_cum = np.cumsum(fp_list).astype(float)
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls    = tp_cum / n_gt

    # All-points AP interpolation (COCO-style)
    mrec = np.concatenate(([0.0], recalls, [recalls[-1]]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return ap, float(precisions[-1]), float(recalls[-1])


def evaluate_model_on_val(model, val_samples, fixed_sigma_by_class,
                          threshold=0.5):
    """
    Run full detection evaluation on val set using the CURRENT model state.

    Returns dict with per-class and overall metrics:
      {class_id: {images, instances, n_preds, P, R, mAP50, mAP50_95}}
    """
    model.eval()
    class_preds  = {0: [], 1: [], 2: []}
    class_gts    = {0: {}, 1: {}, 2: {}}
    class_n_gt   = {0: 0, 1: 0, 2: 0}
    class_n_img  = {0: 0, 1: 0, 2: 0}
    class_n_pred = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for img_idx, sample in enumerate(val_samples):
            probe_sub, _, _ = load_and_preprocess(sample["img_path"])
            H, W = probe_sub.shape
            spots = parse_yolo_to_spots(sample["label_path"], W, H)

            if len(spots) == 0:
                continue

            cls = spots[0]["cls"]
            if cls not in CLASS_NAMES:
                continue

            class_n_img[cls] += 1
            gt_boxes = [(s["cx_px"], s["cy_px"], s["w_px"], s["h_px"]) for s in spots]
            class_gts[cls][img_idx] = gt_boxes
            class_n_gt[cls] += len(gt_boxes)

            # Forward pass through current model
            inp_tensor = torch.from_numpy(probe_sub[np.newaxis, np.newaxis]).float().to(device)
            out_tensor = model(inp_tensor)
            pred_map = out_tensor.squeeze().cpu().numpy()

            # Detect spots from predicted density map
            detected = detect_spots_from_density(pred_map, threshold=threshold)

            for (px, py, conf) in detected:
                class_preds[cls].append((conf, px, py, img_idx))
            class_n_pred[cls] += len(detected)

    # Sort predictions by confidence (descending)
    for cls in class_preds:
        class_preds[cls].sort(key=lambda x: -x[0])

    # Compute metrics per class
    IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
    metrics = {}

    for cls in [0, 1, 2]:
        if class_n_gt[cls] == 0:
            metrics[cls] = dict(images=0, instances=0, n_preds=0,
                                P=0.0, R=0.0, mAP50=0.0, mAP50_95=0.0)
            continue

        ap50, p50, r50 = compute_ap_at_iou(
            class_preds[cls], class_gts[cls], class_n_gt[cls], 0.5)

        aps = []
        for iou_t in IOU_THRESHOLDS:
            ap_t, _, _ = compute_ap_at_iou(
                class_preds[cls], class_gts[cls], class_n_gt[cls], iou_t)
            aps.append(ap_t)

        metrics[cls] = dict(
            images=class_n_img[cls],
            instances=class_n_gt[cls],
            n_preds=class_n_pred[cls],
            P=p50, R=r50,
            mAP50=ap50,
            mAP50_95=float(np.mean(aps))
        )

    # Overall (macro-average over active classes)
    active = [m for m in metrics.values() if m["instances"] > 0]
    if active:
        metrics["overall"] = dict(
            images=sum(m["images"] for m in active),
            instances=sum(m["instances"] for m in active),
            n_preds=sum(m["n_preds"] for m in active),
            P=float(np.mean([m["P"] for m in active])),
            R=float(np.mean([m["R"] for m in active])),
            mAP50=float(np.mean([m["mAP50"] for m in active])),
            mAP50_95=float(np.mean([m["mAP50_95"] for m in active])),
        )
    else:
        metrics["overall"] = dict(images=0, instances=0, n_preds=0,
                                  P=0.0, R=0.0, mAP50=0.0, mAP50_95=0.0)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print(f"  U-FISH FROM-SCRATCH TRAINING — FOLD {VAL_FOLD}")
    print(f"  Validation: fold-{VAL_FOLD}")
    print(f"  Training  : folds {[j for j in range(1, N_FOLDS+1) if j != VAL_FOLD]}")
    print("=" * 75)
    print(f"  Device     : {device}")
    print(f"  LR         : {LR}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 75)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Discover all fold data ─────────────────────────────────────────
    print("\n[1/6] Discovering dataset...")
    fold_samples = {}
    for k in range(1, N_FOLDS + 1):
        fold_name = FOLD_FMT.format(k=k)
        fold_samples[k] = discover_samples(FOLD_ROOT, fold_name)
        print(f"  fold-{k}: {len(fold_samples[k])} images")

    # ── 2. Compute per-class fixed σ ──────────────────────────────────────
    print("\n[2/6] Computing per-class fixed σ...")
    fixed_sigma_by_class = compute_fixed_sigma_by_class(fold_samples, N_FOLDS)
    for c in [0, 1, 2]:
        print(f"  {CLASS_NAMES[c]:>5} : σ = {fixed_sigma_by_class[c]:.2f} px")

    # ── 3. Build train/val splits ─────────────────────────────────────────
    print(f"\n[3/6] Building train/val split (val=fold-{VAL_FOLD})...")
    train_folds = [j for j in range(1, N_FOLDS + 1) if j != VAL_FOLD]
    train_samples = []
    for j in train_folds:
        train_samples.extend(fold_samples[j])
    val_samples = fold_samples[VAL_FOLD]

    print(f"  Train: folds {train_folds} → {len(train_samples)} images")
    print(f"  Val  : fold-{VAL_FOLD}      → {len(val_samples)} images")

    train_ds = FISHDataset(train_samples,
                           fixed_sigma_by_class=fixed_sigma_by_class,
                           augment=True)
    val_ds   = FISHDataset(val_samples,
                           fixed_sigma_by_class=fixed_sigma_by_class,
                           augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    # ── 4. Build model (FROM SCRATCH — random Kaiming-He init) ────────────
    print("\n[4/6] Initialising UFishNet from scratch (random Kaiming-He)...")
    ufish = UFish()
    ufish.init_model()  # Random Kaiming-He initialisation — NO pretrained weights
    model = ufish.model
    assert model is not None, "UFish model was not initialized!"
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model  : UFishNet ({n_params:,} parameters)")
    print(f"  Weights: Random Kaiming-He init (from scratch)")

    # ── 5. Optimizer & Loss ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = DiceRMSELoss().to(device)
    dice_fn   = DiceLoss().to(device)
    rmse_fn   = RMSELoss().to(device)

    print(f"  Optimizer: Adam (lr={LR}, no weight decay, no scheduler)")
    print(f"  Loss     : DiceRMSELoss = 0.6×Dice + 0.4×RMSE")

    # ── Save training config ─────────────────────────────────────────────
    config = {
        "val_fold": VAL_FOLD,
        "train_folds": train_folds,
        "n_folds": N_FOLDS,
        "fold_root": FOLD_ROOT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": LR,
        "optimizer": "Adam (no weight decay, no scheduler)",
        "loss": "DiceRMSELoss = 0.6*Dice + 0.4*RMSE",
        "weight_init": "Random Kaiming-He (from scratch)",
        "n_train_images": len(train_samples),
        "n_val_images": len(val_samples),
        "n_params": n_params,
        "fixed_sigma_by_class": {str(k): v for k, v in fixed_sigma_by_class.items()},
        "device": str(device),
        "started_at": datetime.now().isoformat(),
        "augmentation": "horizontal + vertical flip",
        "intensity_threshold": INTENSITY_THRESHOLD,
    }
    with open(os.path.join(RESULTS_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── 6. Training loop ─────────────────────────────────────────────────
    print(f"\n[5/6] Starting training ({EPOCHS} epochs)...")
    print("=" * 75)

    best_val_loss = float("inf")
    best_epoch    = 0
    best_state    = None

    # CSV header for per-epoch metrics
    csv_path = os.path.join(RESULTS_DIR, "metrics_per_epoch.csv")
    csv_fields = [
        "epoch", "train_loss", "train_dice_loss", "train_rmse_loss",
        "val_loss", "val_dice_loss", "val_rmse_loss",
        "is_best", "epoch_time_sec",
        # Per-class detection metrics
        "green_P", "green_R", "green_mAP50", "green_mAP50_95",
        "green_instances", "green_preds",
        "red_P", "red_R", "red_mAP50", "red_mAP50_95",
        "red_instances", "red_preds",
        "aqua_P", "aqua_R", "aqua_mAP50", "aqua_mAP50_95",
        "aqua_instances", "aqua_preds",
        "overall_P", "overall_R", "overall_mAP50", "overall_mAP50_95",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    all_epoch_data = []
    total_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        running_loss      = 0.0
        running_dice_loss = 0.0
        running_rmse_loss = 0.0
        n_batch = 0

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()

            running_loss      += loss.item()
            running_dice_loss += dice_fn(out, tgt).item()
            running_rmse_loss += rmse_fn(out, tgt).item()
            n_batch += 1

        ep_train_loss = running_loss      / max(n_batch, 1)
        ep_train_dice = running_dice_loss / max(n_batch, 1)
        ep_train_rmse = running_rmse_loss / max(n_batch, 1)

        # ── VALIDATE ─────────────────────────────────────────────────────
        model.eval()
        running_val       = 0.0
        running_val_dice  = 0.0
        running_val_rmse  = 0.0
        n_val = 0

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                running_val      += loss_fn(out, tgt).item()
                running_val_dice += dice_fn(out, tgt).item()
                running_val_rmse += rmse_fn(out, tgt).item()
                n_val += 1

        ep_val_loss = running_val      / max(n_val, 1)
        ep_val_dice = running_val_dice / max(n_val, 1)
        ep_val_rmse = running_val_rmse / max(n_val, 1)

        # ── CHECK BEST ───────────────────────────────────────────────────
        is_best = False
        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            is_best = True

        # ── SAVE MODEL CHECKPOINT (every epoch) ─────────────────────────
        ckpt_path = os.path.join(RESULTS_DIR, f"epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # If this is the best epoch so far, also save as best_model.pth
        if is_best:
            best_path = os.path.join(RESULTS_DIR, "best_model.pth")
            torch.save(best_state, best_path)

        # ── DETECTION EVALUATION ─────────────────────────────────────────
        det_metrics = evaluate_model_on_val(model, val_samples,
                                            fixed_sigma_by_class,
                                            threshold=INTENSITY_THRESHOLD)

        epoch_time = time.time() - t_ep

        # ── BUILD EPOCH RECORD ───────────────────────────────────────────
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(ep_train_loss, 8),
            "train_dice_loss": round(ep_train_dice, 8),
            "train_rmse_loss": round(ep_train_rmse, 8),
            "val_loss": round(ep_val_loss, 8),
            "val_dice_loss": round(ep_val_dice, 8),
            "val_rmse_loss": round(ep_val_rmse, 8),
            "is_best": is_best,
            "epoch_time_sec": round(epoch_time, 2),
            # Green (class 0)
            "green_P": round(det_metrics[0]["P"], 6),
            "green_R": round(det_metrics[0]["R"], 6),
            "green_mAP50": round(det_metrics[0]["mAP50"], 6),
            "green_mAP50_95": round(det_metrics[0]["mAP50_95"], 6),
            "green_instances": det_metrics[0]["instances"],
            "green_preds": det_metrics[0]["n_preds"],
            # Red (class 1)
            "red_P": round(det_metrics[1]["P"], 6),
            "red_R": round(det_metrics[1]["R"], 6),
            "red_mAP50": round(det_metrics[1]["mAP50"], 6),
            "red_mAP50_95": round(det_metrics[1]["mAP50_95"], 6),
            "red_instances": det_metrics[1]["instances"],
            "red_preds": det_metrics[1]["n_preds"],
            # Aqua (class 2)
            "aqua_P": round(det_metrics[2]["P"], 6),
            "aqua_R": round(det_metrics[2]["R"], 6),
            "aqua_mAP50": round(det_metrics[2]["mAP50"], 6),
            "aqua_mAP50_95": round(det_metrics[2]["mAP50_95"], 6),
            "aqua_instances": det_metrics[2]["instances"],
            "aqua_preds": det_metrics[2]["n_preds"],
            # Overall
            "overall_P": round(det_metrics["overall"]["P"], 6),
            "overall_R": round(det_metrics["overall"]["R"], 6),
            "overall_mAP50": round(det_metrics["overall"]["mAP50"], 6),
            "overall_mAP50_95": round(det_metrics["overall"]["mAP50_95"], 6),
        }

        all_epoch_data.append(epoch_record)

        # ── APPEND TO CSV (flush immediately — no waiting) ───────────────
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(epoch_record)

        # ── SAVE RUNNING JSON SUMMARY (overwrite each epoch) ─────────────
        summary = {
            "val_fold": VAL_FOLD,
            "train_folds": train_folds,
            "current_epoch": epoch,
            "total_epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 8),
            "elapsed_min": round((time.time() - total_t0) / 60.0, 2),
            "config": config,
            "epochs": all_epoch_data,
        }
        with open(os.path.join(RESULTS_DIR, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # ── PRINT PROGRESS ───────────────────────────────────────────────
        mark = " ★ BEST" if is_best else ""
        overall = det_metrics["overall"]
        print(f"  Ep {epoch:>3}/{EPOCHS}  "
              f"train={ep_train_loss:.6f}  val={ep_val_loss:.6f}  "
              f"P={overall['P']:.4f}  R={overall['R']:.4f}  "
              f"mAP50={overall['mAP50']:.4f}  mAP50-95={overall['mAP50_95']:.4f}  "
              f"({epoch_time:.1f}s){mark}")

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - total_t0
    print("\n" + "=" * 75)
    print(f"  TRAINING COMPLETE — Fold {VAL_FOLD}")
    print(f"  Total time  : {total_time/60:.1f} min")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best model  : {os.path.join(RESULTS_DIR, 'best_model.pth')}")
    print(f"  Metrics CSV : {csv_path}")
    print("=" * 75)

    # ── Final summary JSON ───────────────────────────────────────────────
    final_summary = {
        "val_fold": VAL_FOLD,
        "train_folds": train_folds,
        "total_epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 8),
        "total_time_sec": round(total_time, 2),
        "total_time_min": round(total_time / 60.0, 2),
        "completed_at": datetime.now().isoformat(),
        "config": config,
        "epochs": all_epoch_data,
    }
    with open(os.path.join(RESULTS_DIR, "training_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    # Cleanup
    del model, optimizer, ufish, loss_fn, dice_fn, rmse_fn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  All results saved to: {RESULTS_DIR}/")
    print(f"  Files:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"    {fname}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
