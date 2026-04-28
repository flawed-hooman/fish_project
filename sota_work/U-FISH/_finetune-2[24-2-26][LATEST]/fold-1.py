#!/usr/bin/env python3
"""
U-FISH Finetuning — Fold 1 (Validate on fold-1, Train on folds 2-5)
====================================================================

Strictly follows the official U-FISH preprocessing and training:
  - Input: probe channel extracted from RGB → scale_image() → [0, 255] float32
  - Target: binary point mask → scipy gaussian_filter(sigma=1) → peak-normalized
  - Loss: DiceRMSELoss = 0.6 × Dice + 0.4 × RMSE
  - Optimizer: Adam, lr=1e-4 (finetuning from pretrained)
  - Architecture: UFishNet (162,959 params)

Dataset: Gene-Dataset-5Fold-Merged (YOLO format, 512×512 RGB)
Ground truth: YOLO bounding boxes → only center (xc, yc) used as point coords
Results saved to: results-1/ (epoch_XXX.pth, best_model.pth, metrics.json)

Reference: U-FISH (Xu et al., Genome Biology 2025)
"""

import copy
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from ufish.api import UFish
from ufish.utils.img import scale_image

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FOLD_ID       = 1                   # This script: validate on fold-1
N_FOLDS       = 5
FOLD_ROOT     = "/home/sukrit/saaransh/fish_project_sota_work/Model_B/Gene-Dataset-5Fold-Merged"
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results-{FOLD_ID}")

EPOCHS        = 50
BATCH_SIZE    = 4
NUM_WORKERS   = 2
LR            = 1e-4               # Finetuning LR (lower to avoid catastrophic forgetting)
SIGMA         = 1.0                 # Official U-FISH Gaussian sigma for target density
SUBTRACT_DAPI = True                # Subtract DAPI bleed-through from non-blue probes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print(f"  U-FISH FINETUNING — FOLD {FOLD_ID}")
print(f"  Validate: fold-{FOLD_ID}   Train: folds {[k for k in range(1, N_FOLDS+1) if k != FOLD_ID]}")
print(f"  Epochs: {EPOCHS}   BS: {BATCH_SIZE}   LR: {LR}   σ: {SIGMA}")
print(f"  DAPI subtraction: {SUBTRACT_DAPI}")
print(f"  Results: {RESULTS_DIR}")
print(f"  Device: {DEVICE}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. YOLO → Point Coordinates
# ═══════════════════════════════════════════════════════════════════════════════

def parse_yolo_to_points(label_path, img_w, img_h):
    """Parse YOLO labels → list of (row, col) pixel coordinates.

    U-FISH uses POINT annotations. From YOLO boxes we take ONLY the center.
    Box dimensions are NOT used for training — only for evaluation later.
    """
    points = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc = float(parts[1]) * img_w   # center x in pixels
            yc = float(parts[2]) * img_h   # center y in pixels
            points.append((yc, xc))         # (row, col) convention
    return points


def parse_yolo_boxes(label_path, img_w, img_h):
    """Parse YOLO labels → list of (xc, yc, w, h) in pixels. For evaluation."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            w  = float(parts[3]) * img_w
            h  = float(parts[4]) * img_h
            boxes.append((xc, yc, w, h))
    return boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Target Density Map — OFFICIAL U-FISH METHOD
# ═══════════════════════════════════════════════════════════════════════════════
#
# Source: ufish/data.py → FISHSpotsDataset.coords_to_target + gaussian_filter
#
# Algorithm:
#   1. Binary point mask: zeros with 1 at each spot coordinate
#   2. scipy.ndimage.gaussian_filter(mask, sigma=1)
#   3. Peak-normalize: divide by min peak value → all spots reach ≥ 1.0
#
# This is the EXACT same method used to train the pretrained U-FISH model.

def make_density_map_official(points, img_h, img_w, sigma=1.0):
    """Generate density map using the official U-FISH method.

    Args:
        points: list of (row, col) tuples — spot coordinates
        img_h, img_w: image dimensions
        sigma: Gaussian sigma (1.0 = official U-FISH default)

    Returns:
        density: float32 array (H, W), peaks ≈ 1.0
    """
    mask = np.zeros((img_h, img_w), dtype=np.float32)

    if len(points) == 0:
        return mask

    # Step 1: Place 1s at each spot coordinate
    valid_coords = []
    for r, c in points:
        ri = int(round(r))
        ci = int(round(c))
        ri = max(0, min(ri, img_h - 1))
        ci = max(0, min(ci, img_w - 1))
        mask[ri, ci] = 1.0
        valid_coords.append((ri, ci))

    if len(valid_coords) == 0:
        return mask

    # Step 2: Gaussian filter (official uses sigma=1)
    density = ndi.gaussian_filter(mask, sigma=sigma)

    # Step 3: Peak normalization (official method)
    # Read the value at each spot center after filtering
    peak_vals = np.array([density[r, c] for r, c in valid_coords])

    if peak_vals.min() > 0:
        density = density / peak_vals.min()

    return density


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Preprocessing — OFFICIAL U-FISH scale_image()
# ═══════════════════════════════════════════════════════════════════════════════

def get_probe_channel(filename):
    """Detect probe channel from filename: ORANGE→R(0), FITC→G(1), AQUA→B(2)."""
    name = os.path.basename(filename).upper()
    if "ORANGE" in name:
        return 0
    if "FITC" in name or "GREEN" in name:
        return 1
    if "AQUA" in name:
        return 2
    return 0  # default Red


def load_and_preprocess(img_path, subtract_dapi=True):
    """Load RGB → extract probe channel → optional DAPI subtract → scale_image [0,255].

    This matches U-FISH's expected input format:
      - Single channel, float32, scaled to [0, 255] via scale_image()
      - scale_image: percentile-based outlier clipping + rescale_intensity
    """
    img_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    img_f = img_rgb.astype(np.float32) / 255.0

    probe_ch = get_probe_channel(str(img_path))
    probe = img_f[:, :, probe_ch]

    if subtract_dapi and probe_ch != 2:
        # Subtract DAPI bleed-through (Blue channel)
        dapi = img_f[:, :, 2]
        probe = np.clip(probe - dapi, 0.0, 1.0)

    # Convert to [0, 255] range then apply official scale_image()
    probe_255 = (probe * 255.0).astype(np.float32)
    scaled = scale_image(probe_255, warning=False)

    return scaled, img_rgb, probe_ch


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Loss Functions — EXACT U-FISH Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = (pred ** 2).sum() + (target ** 2).sum()
        return 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)


class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(nn.functional.mse_loss(pred, target))


class DiceRMSELoss(nn.Module):
    """U-FISH loss: 0.6 × Dice + 0.4 × RMSE."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.rmse = RMSELoss()

    def forward(self, pred, target):
        return 0.6 * self.dice(pred, target) + 0.4 * self.rmse(pred, target)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FISHDataset(Dataset):
    """U-FISH finetuning dataset.

    Input:  probe channel → scale_image() → (1, H, W) float32 in [0, 255]
    Target: official density map (σ=1, peak-normalized) → (1, H, W) float32
    """

    def __init__(self, samples, sigma=1.0, subtract_dapi=True, augment=True):
        self.samples = samples
        self.sigma = sigma
        self.subtract_dapi = subtract_dapi
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Preprocess input (probe channel, [0, 255] via scale_image)
        scaled_input, _, _ = load_and_preprocess(s["img_path"], self.subtract_dapi)
        h, w = scaled_input.shape

        # Parse YOLO → point coordinates (center only, no box dims)
        points = parse_yolo_to_points(s["label_path"], w, h)

        # Generate target density map (official method, σ=1)
        density = make_density_map_official(points, h, w, sigma=self.sigma)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                scaled_input = np.flip(scaled_input, axis=1).copy()
                density = np.flip(density, axis=1).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                scaled_input = np.flip(scaled_input, axis=0).copy()
                density = np.flip(density, axis=0).copy()
            # Random 90° rotation
            k = np.random.randint(0, 4)
            if k > 0:
                scaled_input = np.rot90(scaled_input, k=k).copy()
                density = np.rot90(density, k=k).copy()

        inp = torch.from_numpy(scaled_input[np.newaxis].copy()).float()  # (1,H,W)
        tgt = torch.from_numpy(density[np.newaxis].copy()).float()       # (1,H,W)
        return inp, tgt


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Discover Samples
# ═══════════════════════════════════════════════════════════════════════════════

def discover_samples(fold_root, fold_name):
    """Find image+label pairs in a fold directory."""
    img_dir = os.path.join(fold_root, fold_name, "images")
    label_dir = os.path.join(fold_root, fold_name, "labels")

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )

    results = []
    skipped = 0
    for img_path in img_paths:
        stem = Path(img_path).stem
        label_path = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(label_path):
            results.append({"img_path": img_path, "label_path": label_path})
        else:
            skipped += 1

    if skipped:
        print(f"  Warning: {skipped} images skipped (no label)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    # Discover all fold samples
    print("\n--- Discovering dataset ---")
    fold_samples = {}
    for k in range(1, N_FOLDS + 1):
        fold_name = f"fold-{k}"
        fold_samples[k] = discover_samples(FOLD_ROOT, fold_name)
        print(f"  fold-{k}: {len(fold_samples[k])} samples")

    # Build train/val splits
    val_samples = fold_samples[FOLD_ID]
    train_samples = []
    for k in range(1, N_FOLDS + 1):
        if k != FOLD_ID:
            train_samples.extend(fold_samples[k])

    print(f"\n  Train: {len(train_samples)} images (folds {[k for k in range(1, N_FOLDS+1) if k != FOLD_ID]})")
    print(f"  Val:   {len(val_samples)} images (fold-{FOLD_ID})")

    # Create datasets
    train_ds = FISHDataset(train_samples, sigma=SIGMA,
                           subtract_dapi=SUBTRACT_DAPI, augment=True)
    val_ds = FISHDataset(val_samples, sigma=SIGMA,
                         subtract_dapi=SUBTRACT_DAPI, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Load pretrained U-FISH model
    print("\n--- Loading pretrained U-FISH model ---")
    uf = UFish()
    uf.load_weights(weights_file="v1.0-alldata-ufish_c32.pth")
    model = uf.model
    assert model is not None, "Model not loaded!"
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UFishNet: {n_params:,} parameters")
    print(f"  Pretrained weights loaded (Nature Methods 2024)")

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceRMSELoss().to(DEVICE)

    # Training state
    best_val_loss = float("inf")
    best_epoch = -1
    all_metrics = []

    print(f"\n--- Training ---")
    total_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_dice_losses = []
        train_rmse_losses = []
        dice_fn = DiceLoss().to(DEVICE)
        rmse_fn = RMSELoss().to(DEVICE)

        for batch_inp, batch_tgt in train_loader:
            batch_inp = batch_inp.to(DEVICE)
            batch_tgt = batch_tgt.to(DEVICE)

            optimizer.zero_grad()
            pred = model(batch_inp)
            loss = criterion(pred, batch_tgt)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                d_loss = dice_fn(pred, batch_tgt).item()
                r_loss = rmse_fn(pred, batch_tgt).item()

            train_losses.append(loss.item())
            train_dice_losses.append(d_loss)
            train_rmse_losses.append(r_loss)

        avg_train      = float(np.mean(train_losses))
        avg_train_dice = float(np.mean(train_dice_losses))
        avg_train_rmse = float(np.mean(train_rmse_losses))

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_dice_losses = []
        val_rmse_losses = []

        with torch.no_grad():
            for batch_inp, batch_tgt in val_loader:
                batch_inp = batch_inp.to(DEVICE)
                batch_tgt = batch_tgt.to(DEVICE)

                pred = model(batch_inp)
                loss = criterion(pred, batch_tgt)

                d_loss = dice_fn(pred, batch_tgt).item()
                r_loss = rmse_fn(pred, batch_tgt).item()

                val_losses.append(loss.item())
                val_dice_losses.append(d_loss)
                val_rmse_losses.append(r_loss)

        avg_val      = float(np.mean(val_losses))
        avg_val_dice = float(np.mean(val_dice_losses))
        avg_val_rmse = float(np.mean(val_rmse_losses))

        elapsed = time.time() - epoch_t0

        # ── Save epoch checkpoint ────────────────────────────────────────
        epoch_path = os.path.join(RESULTS_DIR, f"epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), epoch_path)

        # ── Check best ───────────────────────────────────────────────────
        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_path = os.path.join(RESULTS_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            improved = " ★ BEST"

        # ── Log metrics ──────────────────────────────────────────────────
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train,
            "train_dice": avg_train_dice,
            "train_rmse": avg_train_rmse,
            "val_loss": avg_val,
            "val_dice": avg_val_dice,
            "val_rmse": avg_val_rmse,
            "epoch_time_sec": elapsed,
            "is_best": bool(improved),
            "checkpoint": os.path.basename(epoch_path),
        }
        all_metrics.append(epoch_metrics)

        # Print progress
        print(f"  Ep {epoch:3d}/{EPOCHS}  "
              f"train={avg_train:.5f} (D:{avg_train_dice:.4f} R:{avg_train_rmse:.4f})  "
              f"val={avg_val:.5f} (D:{avg_val_dice:.4f} R:{avg_val_rmse:.4f})  "
              f"[{elapsed:.1f}s]{improved}")

        # Save metrics JSON after every epoch (so we have partial results if interrupted)
        _save_metrics(all_metrics, best_epoch, best_val_loss, total_t0)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"  FOLD {FOLD_ID} COMPLETE — {total_time / 60:.1f} min")
    print(f"  Best val loss: {best_val_loss:.5f} @ epoch {best_epoch}")
    print(f"  Best model: {RESULTS_DIR}/best_model.pth")
    print(f"{'=' * 70}")


def _save_metrics(all_metrics, best_epoch, best_val_loss, total_t0):
    """Save training metrics JSON (called after every epoch)."""
    results = {
        "fold_id": FOLD_ID,
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "sigma": SIGMA,
            "subtract_dapi": SUBTRACT_DAPI,
            "method": "U-FISH official finetuning (σ=1, scale_image, DiceRMSE)",
            "pretrained": True,
            "val_fold": FOLD_ID,
            "train_folds": [k for k in range(1, N_FOLDS + 1) if k != FOLD_ID],
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_sec": time.time() - total_t0,
        "epochs": all_metrics,
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()
