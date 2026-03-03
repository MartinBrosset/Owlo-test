"""
train_functions.py
------------------
Dataset, losses, and training utilities for the cell segmentation pipeline.

Data layout
-----------
  vol_linear_{k}.tif  – multi-page TIFF, shape (61, 256, 255), uint8
  mask_{k}.tif        – same shape; pixel values ∈ {0, 51, 102, 153, 204, 255}
                        encoding = label_index × 51  (labels 0-5)

Class remapping (version 1)
---------------------------
  raw label 0, 5          -> class 0  (background)
  raw labels 1, 2         -> class 1  (cell type A)
  raw labels 3, 4         -> class 2  (cell type B)

Class imbalance (approx. for mask_0)
-------------------------------------
  class 0 ~80 %,  class 1 ~20 %,  class 2 ~1 %
  -> use CombinedLoss (Focal + Dice) to handle imbalance robustly.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageSequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_CMAP = mcolors.ListedColormap(["black", "deepskyblue", "tomato"])


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tif_stack(path: str) -> np.ndarray:
    """Load every frame of a multi-page TIFF -> (N, H, W) uint8 array."""
    img = Image.open(path)
    frames = [np.array(f.copy()) for f in ImageSequence.Iterator(img)]
    return np.stack(frames, axis=0)


def process_mask(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw uint8 mask (values in {0,51,102,153,204,255}) to class indices.

    Returns int64 array with values in {0, 1, 2}.
    """
    labels = (raw.astype(np.int64)) // 51   # -> {0,1,2,3,4,5}
    out = np.zeros_like(labels)
    out[(labels == 1) | (labels == 2)] = 1
    out[(labels == 3) | (labels == 4)] = 2
    return out                               # background = 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CellDataset(Dataset):
    """
    Each sample is one 2D slice (image, mask).

    Args
    ----
    data_dir        : folder containing vol_linear_{k}.tif and mask_{k}.tif
    volume_indices  : which k values to include
    img_size        : (H, W) to resize to, or None to keep native (256×255)
    augment         : if True apply random horizontal / vertical flips
    """

    def __init__(
        self,
        data_dir: str,
        volume_indices: list[int],
        img_size: Optional[tuple] = None,
        augment: bool = False,
        n_slices: Optional[int] = None,
        window_frac: float = 0.4,
        seed: int = 0,
        stride: int = 1,
    ):
        """
        n_slices    : randomly sample this many slices per volume from a central
                      window (None = use all slices, or every `stride`-th if stride>1)
        window_frac : fraction of depth kept around the middle for random sampling
        seed        : for reproducibility of the per-volume random sample
        stride      : take every Nth slice (e.g. stride=2 halves the data).
                      Only used when n_slices is None.
        """
        self.img_size = img_size
        self.augment  = augment
        self.images: list[np.ndarray] = []   # float32, shape (H, W)
        self.masks:  list[np.ndarray] = []   # int64,   shape (H, W)

        rng = np.random.default_rng(seed)
        loaded = 0
        for k in volume_indices:
            img_path  = os.path.join(data_dir, f"vol_linear_{k}.tif")
            mask_path = os.path.join(data_dir, f"mask_{k}.tif")
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"  [warn] skipping k={k}: file not found")
                continue

            img_stack  = load_tif_stack(img_path).astype(np.float32) / 255.0
            mask_stack = load_tif_stack(mask_path)

            n_total = img_stack.shape[0]
            if n_slices is None:
                slice_ids = list(range(0, n_total, stride))
            else:
                half_win  = int(n_total * window_frac / 2)
                mid       = n_total // 2
                lo        = max(0, mid - half_win)
                hi        = min(n_total, mid + half_win)
                available = np.arange(lo, hi)
                chosen    = rng.choice(available, size=min(n_slices, len(available)), replace=False)
                slice_ids = sorted(chosen.tolist())

            for s in slice_ids:
                self.images.append(img_stack[s])
                self.masks.append(process_mask(mask_stack[s]))
            loaded += 1

        print(f"CellDataset: {loaded} volumes -> {len(self.images)} slices")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img  = self.images[idx]   # (H, W) float32
        mask = self.masks[idx]    # (H, W) int64

        img_t  = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask)                # (H, W)

        if self.img_size is not None:
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=self.img_size,
                mode="bilinear", align_corners=False
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(), size=self.img_size,
                mode="nearest"
            ).squeeze().long()

        if self.augment:
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])

        return img_t, mask_t


def make_dataloaders(
    data_dir: str,
    all_indices: list[int],
    val_frac: float = 0.15,
    batch_size: int = 8,
    img_size: Optional[tuple] = None,
    num_workers: int = 0,
    n_slices: Optional[int] = None,
    stride: int = 2,
) -> tuple:
    """
    Shuffle and split indices into train/val; return (train_loader, val_loader).
    stride   : take every Nth slice per volume (default 2 = halve the data).
    n_slices : if set, randomly sample this many slices instead of using stride.
    """
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(all_indices).tolist()
    n_val   = max(1, int(len(indices) * val_frac))

    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = CellDataset(data_dir, train_idx, img_size=img_size, augment=True,  n_slices=n_slices, stride=stride)
    val_ds   = CellDataset(data_dir, val_idx,   img_size=img_size, augment=False, n_slices=n_slices, stride=stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  train slices: {len(train_ds)}  |  val slices: {len(val_ds)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Losses – robust to class imbalance
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    FL(p) = -(1-p_t)^γ · log(p_t)
    Higher γ -> more focus on hard / rare examples.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # per-class weight tensor, passed to nll_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W)
        # targets: (B, H, W) long
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        pt    = torch.exp(-ce)
        loss  = (1 - pt) ** self.gamma * ce
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss, macro-averaged.
    ignore_bg=True excludes background from averaging (recommended).
    """

    def __init__(self, num_classes: int = 3, smooth: float = 1e-6, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                              # (B, C, H, W)
        t_oh  = F.one_hot(targets, self.num_classes)                  # (B, H, W, C)
        t_oh  = t_oh.permute(0, 3, 1, 2).float()                     # (B, C, H, W)

        start = 1 if self.ignore_bg else 0
        dice_terms = []
        for c in range(start, self.num_classes):
            p = probs[:, c].reshape(-1)
            t = t_oh[:, c].reshape(-1)
            intersection = (p * t).sum()
            dice_terms.append(
                1.0 - (2.0 * intersection + self.smooth)
                    / (p.sum() + t.sum() + self.smooth)
            )
        return torch.stack(dice_terms).mean()


class CombinedLoss(nn.Module):
    """
    0.5 × FocalLoss  +  0.5 × DiceLoss

    Both terms handle class imbalance:
      • Focal  -> down-weights easy background pixels
      • Dice   -> class-size invariant, especially good for rare class 2
    """

    def __init__(
        self,
        num_classes: int = 3,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.dice  = DiceLoss(num_classes=num_classes, ignore_bg=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)
        #return self.dice(logits, targets)


def compute_class_weights(
    data_dir: str,
    indices: list[int],
    num_classes: int = 3,
    max_volumes: int = 10,
) -> torch.Tensor:
    """
    Estimate per-class weights from a subset of volumes.
    Uses inverse-frequency weighting: w_c = N_total / (C × N_c).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for k in indices[:max_volumes]:
        path = os.path.join(data_dir, f"mask_{k}.tif")
        if not os.path.exists(path):
            continue
        stack = load_tif_stack(path)
        for frame in stack:
            processed = process_mask(frame)
            for c in range(num_classes):
                counts[c] += (processed == c).sum()

    total   = counts.sum()
    weights = total / (num_classes * counts + 1e-8)
    weights = weights / weights.min()            # normalise so smallest weight = 1
    print(f"Class counts (subset): {counts.astype(int)}")
    print(f"Class weights: {weights.round(3)}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3):
    """
    Per-class IoU.
    preds / targets: integer arrays of arbitrary shape.
    Returns array of shape (num_classes,).
    """
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum()
        union = ((preds == c) | (targets == c)).sum()
        iou[c] = inter / (union + 1e-8)
    return iou


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    epoch: int = 0,
    num_epochs: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n_imgs     = 0

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:3d}/{num_epochs} [train]",
        unit="batch",
        leave=False,
    )
    for imgs, masks in bar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_imgs     += imgs.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}", imgs=n_imgs)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    epoch: int = 0,
    num_epochs: int = 0,
) -> tuple:
    """
    Returns (val_loss, iou_per_class, mean_iou).
    mean_iou is computed over foreground classes only (excludes background).
    """
    model.eval()
    total_loss   = 0.0
    intersection = np.zeros(num_classes)
    union        = np.zeros(num_classes)

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:3d}/{num_epochs} [val]  ",
        unit="batch",
        leave=False,
    )
    for imgs, masks in bar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks).item()
        total_loss += loss

        preds = logits.argmax(dim=1).cpu().numpy()
        gt    = masks.cpu().numpy()
        for c in range(num_classes):
            intersection[c] += ((preds == c) & (gt == c)).sum()
            union[c]        += ((preds == c) | (gt == c)).sum()

        bar.set_postfix(loss=f"{loss:.4f}")

    iou_per_class = intersection / (union + 1e-8)
    mean_iou      = iou_per_class[1:].mean()   # foreground only
    return total_loss / len(loader), iou_per_class, mean_iou


def plot_val_samples(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    n_samples: int = 3,
) -> None:
    """Pick n_samples random slices from val_dataset and show image / GT / prediction."""
    indices = np.random.choice(len(val_dataset), size=min(n_samples, len(val_dataset)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_t, mask_t = val_dataset[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy()

            axes[row, 0].imshow(img_t.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Image  (sample {idx})")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_t.numpy(), cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    fig.suptitle(f"Validation samples — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.show()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,                         # e.g. ReduceLROnPlateau
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 30,
    save_path: Optional[str] = None,
    num_classes: int = 3,
    plot_fn=None,   # callable(model, dataset, device, epoch); defaults to plot_val_samples
) -> dict:
    """
    Full training loop.

    Returns history dict with keys:
        'train_loss', 'val_loss', 'mean_iou', 'iou_per_class'
    """
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    history: dict = {"train_loss": [], "val_loss": [], "mean_iou": [], "iou_per_class": []}
    best_iou = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            epoch=epoch, num_epochs=num_epochs,
        )
        val_loss, iou_per_class, mean_iou = validate(
            model, val_loader, criterion, device, num_classes,
            epoch=epoch, num_epochs=num_epochs,
        )

        if scheduler is not None:
            # ReduceLROnPlateau needs the metric; other schedulers don't
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_iou"].append(mean_iou)
        history["iou_per_class"].append(iou_per_class.tolist())

        iou_str = "  ".join(
            f"c{c}:{iou_per_class[c]:.3f}" for c in range(num_classes)
        )
        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mIoU={mean_iou:.4f}  [{iou_str}]"
        )

        if mean_iou > best_iou and save_path is not None:
            best_iou = mean_iou
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ saved best checkpoint  (mIoU={best_iou:.4f})")

        if epoch % 2 == 0:
            _plot = plot_fn if plot_fn is not None else plot_val_samples
            _plot(model, val_loader.dataset, device, epoch)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_slice(model: nn.Module, img_np: np.ndarray, device: torch.device,
                  img_size: Optional[tuple] = None) -> np.ndarray:
    """
    Run inference on a single 2D grayscale slice (H, W) float32 in [0,1].
    Returns predicted class map (H, W) int64.
    """
    model.eval()
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if img_size is not None:
        img_t = F.interpolate(img_t, size=img_size, mode="bilinear", align_corners=False)
    with torch.no_grad():
        logits = model(img_t.to(device))
    return logits.argmax(dim=1).squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3-D dataset and helpers  (for UNet3D)
# ─────────────────────────────────────────────────────────────────────────────

class CellDataset3D(Dataset):
    """
    Each sample is a 3D patch (1, D, H, W) centred on a chosen slice,
    paired with the center-slice mask (H, W).

    Volumes are loaded once into RAM (shared across items) and patches are
    extracted on-demand in __getitem__ to avoid duplicating data in memory.

    Args
    ----
    data_dir        : folder containing vol_linear_{k}.tif and mask_{k}.tif
    volume_indices  : which k values to include
    depth           : number of consecutive slices per patch (odd recommended)
    img_size        : (H, W) to resize to, or None to keep native size
    augment         : random horizontal / vertical flips
    stride          : take every Nth slice as patch center (default 2)
    """

    def __init__(
        self,
        data_dir: str,
        volume_indices: list,
        depth: int = 5,
        img_size: Optional[tuple] = None,
        augment: bool = False,
        stride: int = 2,
    ):
        self.depth    = depth
        self.img_size = img_size
        self.augment  = augment
        self._vols: list  = []   # list of (img_stack float32 [0,1], mask_stack uint8)
        self._items: list = []   # list of (vol_idx, center_slice_idx)

        loaded = 0
        for k in volume_indices:
            img_path  = os.path.join(data_dir, f"vol_linear_{k}.tif")
            mask_path = os.path.join(data_dir, f"mask_{k}.tif")
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"  [warn] skipping k={k}: file not found")
                continue

            img_stack  = load_tif_stack(img_path).astype(np.float32) / 255.0
            mask_stack = load_tif_stack(mask_path)
            vol_idx    = len(self._vols)
            self._vols.append((img_stack, mask_stack))

            n_total = img_stack.shape[0]
            for center_idx in range(0, n_total, stride):
                self._items.append((vol_idx, center_idx))
            loaded += 1

        print(f"CellDataset3D: {loaded} volumes -> {len(self._items)} patches (depth={depth}, stride={stride})")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        vol_idx, center_idx = self._items[idx]
        img_stack, mask_stack = self._vols[vol_idx]

        half = self.depth // 2
        n    = img_stack.shape[0]
        # Edge padding: clamp out-of-range slice indices to boundary
        slice_ids   = [max(0, min(n - 1, center_idx + d - half)) for d in range(self.depth)]
        img_patch   = np.stack([img_stack[s] for s in slice_ids], axis=0)  # (D, H, W) float32
        center_mask = process_mask(mask_stack[center_idx])                  # (H, W) int64

        img_t  = torch.from_numpy(img_patch).unsqueeze(0)  # (1, D, H, W)
        mask_t = torch.from_numpy(center_mask)              # (H, W)

        if self.img_size is not None:
            # Resize each depth slice independently via a (D,1,H,W) batch
            imgs_2d = img_t.squeeze(0)                          # (D, H, W)
            imgs_2d = F.interpolate(
                imgs_2d.unsqueeze(1), size=self.img_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)                                        # (D, H, W)
            img_t  = imgs_2d.unsqueeze(0)                       # (1, D, H, W)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(), size=self.img_size,
                mode="nearest",
            ).squeeze().long()

        if self.augment:
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])

        return img_t, mask_t


def make_dataloaders_3d(
    data_dir: str,
    all_indices: list,
    val_frac: float = 0.15,
    batch_size: int = 4,
    img_size: Optional[tuple] = None,
    num_workers: int = 0,
    depth: int = 5,
    stride: int = 2,
) -> tuple:
    """
    Shuffle and split indices into train/val; return (train_loader, val_loader)
    for the 3D UNet dataset.

    depth    : number of slices per 3D patch
    stride   : take every Nth slice as patch center (default 2)
    """
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(all_indices).tolist()
    n_val   = max(1, int(len(indices) * val_frac))

    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = CellDataset3D(data_dir, train_idx, depth=depth, img_size=img_size, augment=True,  stride=stride)
    val_ds   = CellDataset3D(data_dir, val_idx,   depth=depth, img_size=img_size, augment=False, stride=stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  train patches: {len(train_ds)}  |  val patches: {len(val_ds)}")
    return train_loader, val_loader


def predict_slice_3d(
    model: nn.Module,
    img_np: np.ndarray,
    device: torch.device,
    img_size: Optional[tuple] = None,
) -> np.ndarray:
    """
    Run inference with a 3D model on a depth patch (D, H, W) float32 in [0,1].
    Returns predicted class map (H, W) int64 for the center slice.
    """
    model.eval()
    imgs_2d = torch.from_numpy(img_np)               # (D, H, W)
    if img_size is not None:
        imgs_2d = F.interpolate(
            imgs_2d.unsqueeze(1), size=img_size,
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    img_t = imgs_2d.unsqueeze(0).unsqueeze(0)        # (1, 1, D, H, W)
    with torch.no_grad():
        logits = model(img_t.to(device))
    return logits.argmax(dim=1).squeeze().cpu().numpy()


def plot_val_samples_3d(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    n_samples: int = 3,
) -> None:
    """
    Pick n_samples random patches from val_dataset and display:
    center slice image | ground-truth mask | predicted mask.
    """
    indices = np.random.choice(len(val_dataset), size=min(n_samples, len(val_dataset)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_t, mask_t = val_dataset[idx]           # (1, D, H, W), (H, W)
            logits = model(img_t.unsqueeze(0).to(device))   # (1, C, H, W)
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy()
            center_slice = img_t[0, img_t.shape[1] // 2].numpy()  # (H, W)

            axes[row, 0].imshow(center_slice, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Center slice  (patch {idx})")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_t.numpy(), cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    fig.suptitle(f"3-D Validation samples — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SAM pre-extracted embedding datasets  (for SAMSeg2D / SAMSeg3D)
# ─────────────────────────────────────────────────────────────────────────────

class CellDatasetSAMEmbed(Dataset):
    """
    2-D dataset that loads pre-extracted SAM embeddings (256, 64, 64) instead
    of raw images, avoiding the heavy 1024×1024 SAM encoder at each step.

    Expects files: {embed_dir}/emb_{k}.npz  with key 'embeddings'
    of shape (N_slices, 256, 64, 64), dtype float16.

    Masks are loaded from {mask_dir}/mask_{k}.tif as usual.
    """

    def __init__(
        self,
        embed_dir: str,
        mask_dir: str,
        volume_indices: list,
        augment: bool = False,
        stride: int = 2,
    ):
        self.augment = augment
        self.embeddings: list = []
        self.masks: list = []

        loaded = 0
        for k in volume_indices:
            emb_path  = os.path.join(embed_dir, f"emb_{k}.npz")
            mask_path = os.path.join(mask_dir,  f"mask_{k}.tif")
            if not os.path.exists(emb_path):
                print(f"  [warn] k={k}: embedding not found – run pre-extraction first")
                continue
            if not os.path.exists(mask_path):
                print(f"  [warn] k={k}: mask not found")
                continue

            embeds     = np.load(emb_path)["embeddings"]  # (N, 256, 64, 64) float16
            mask_stack = load_tif_stack(mask_path)          # (N, H, W)  uint8
            for s in range(0, len(embeds), stride):
                self.embeddings.append(embeds[s])
                self.masks.append(process_mask(mask_stack[s]))
            loaded += 1

        print(f"CellDatasetSAMEmbed: {loaded} volumes -> {len(self.embeddings)} samples (stride={stride})")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        emb_t  = torch.from_numpy(self.embeddings[idx].astype(np.float32))  # (256,64,64)
        mask_t = torch.from_numpy(self.masks[idx])                           # (H, W)

        if self.augment:
            if torch.rand(1) > 0.5:
                emb_t  = torch.flip(emb_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1) > 0.5:
                emb_t  = torch.flip(emb_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
        return emb_t, mask_t


class CellDatasetSAMEmbed3D(Dataset):
    """
    3-D dataset that loads a depth stack of SAM embeddings (D, 256, 64, 64)
    per sample, paired with the center-slice mask.

    Expects the same {embed_dir}/emb_{k}.npz format as CellDatasetSAMEmbed.
    """

    def __init__(
        self,
        embed_dir: str,
        mask_dir: str,
        volume_indices: list,
        depth: int = 5,
        augment: bool = False,
        stride: int = 2,
    ):
        self.depth   = depth
        self.augment = augment
        self._vols: list  = []
        self._items: list = []

        loaded = 0
        for k in volume_indices:
            emb_path  = os.path.join(embed_dir, f"emb_{k}.npz")
            mask_path = os.path.join(mask_dir,  f"mask_{k}.tif")
            if not os.path.exists(emb_path):
                print(f"  [warn] k={k}: embedding not found – run pre-extraction first")
                continue
            if not os.path.exists(mask_path):
                continue

            embeds     = np.load(emb_path)["embeddings"]  # (N, 256, 64, 64) float16
            mask_stack = load_tif_stack(mask_path)
            vol_idx    = len(self._vols)
            self._vols.append((embeds, mask_stack))
            for centre in range(0, len(embeds), stride):
                self._items.append((vol_idx, centre))
            loaded += 1

        print(f"CellDatasetSAMEmbed3D: {loaded} volumes -> {len(self._items)} patches (depth={depth}, stride={stride})")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        vol_idx, centre = self._items[idx]
        embeds, mask_stack = self._vols[vol_idx]

        half = self.depth // 2
        n    = len(embeds)
        ids  = [max(0, min(n - 1, centre + d - half)) for d in range(self.depth)]
        emb_stack   = np.stack([embeds[s].astype(np.float32) for s in ids], axis=0)  # (D,256,64,64)
        centre_mask = process_mask(mask_stack[centre])                                 # (H, W)

        emb_t  = torch.from_numpy(emb_stack)    # (D, 256, 64, 64)
        mask_t = torch.from_numpy(centre_mask)  # (H, W)

        if self.augment:
            if torch.rand(1) > 0.5:
                emb_t  = torch.flip(emb_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1) > 0.5:
                emb_t  = torch.flip(emb_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
        return emb_t, mask_t


def make_dataloaders_sam_embed(
    embed_dir: str,
    mask_dir: str,
    all_indices: list,
    val_frac: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 0,
    stride: int = 2,
) -> tuple:
    """Train/val split for SAM 2-D embed datasets. Batch size can be large
    since the encoder is not called during training."""
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(all_indices).tolist()
    n_val   = max(1, int(len(indices) * val_frac))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = CellDatasetSAMEmbed(embed_dir, mask_dir, train_idx, augment=True,  stride=stride)
    val_ds   = CellDatasetSAMEmbed(embed_dir, mask_dir, val_idx,   augment=False, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"  train: {len(train_ds)}  |  val: {len(val_ds)}")
    return train_loader, val_loader


def make_dataloaders_sam_embed_3d(
    embed_dir: str,
    mask_dir: str,
    all_indices: list,
    val_frac: float = 0.15,
    batch_size: int = 8,
    depth: int = 5,
    stride: int = 2,
    num_workers: int = 0,
) -> tuple:
    """Train/val split for SAM 3-D embed datasets."""
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(all_indices).tolist()
    n_val   = max(1, int(len(indices) * val_frac))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = CellDatasetSAMEmbed3D(embed_dir, mask_dir, train_idx, depth=depth, augment=True,  stride=stride)
    val_ds   = CellDatasetSAMEmbed3D(embed_dir, mask_dir, val_idx,   depth=depth, augment=False, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"  train patches: {len(train_ds)}  |  val patches: {len(val_ds)}")
    return train_loader, val_loader


def plot_val_samples_sam_embed(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    n_samples: int = 3,
) -> None:
    """
    Validation plot for SAM embedding datasets.
    Shows the mean of the 256 embedding channels as a proxy image
    (since no raw image is stored), plus GT mask and prediction.
    """
    indices = np.random.choice(len(val_dataset), size=min(n_samples, len(val_dataset)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            emb_t, mask_t = val_dataset[idx]           # (256,64,64) or (D,256,64,64)
            logits = model(emb_t.unsqueeze(0).to(device))
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy()

            # Mean over channel dim as proxy visualisation
            if emb_t.ndim == 3:
                proxy = emb_t.mean(0).numpy()                         # (64, 64)
            else:
                proxy = emb_t[emb_t.shape[0] // 2].mean(0).numpy()   # centre slice mean

            axes[row, 0].imshow(proxy, cmap="gray")
            axes[row, 0].set_title(f"SAM emb. mean  (sample {idx})")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_t.numpy(), cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap=_CMAP, vmin=0, vmax=2)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    fig.suptitle(f"SAM Embed Validation — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.show()
