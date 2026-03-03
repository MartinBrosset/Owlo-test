"""
view_volume.py
--------------
Interactive 3D viewer for a vol_linear_{k}.tif + mask_{k}.tif pair.
Navigate slices with  j (previous)  /  k (next).
"""

import importlib.util
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ────────────────────────────────────────────────────────────────────
OWLO_DIR = r"C:\Users\marti\Desktop\Owlo-test"
DATA_DIR = r"C:\Users\marti\Downloads\2025-11-19_Simu_twocells\2025-11-19_Simu_twocells"
sys.path.insert(0, OWLO_DIR)

# ── load helpers ─────────────────────────────────────────────────────────────
from train_functions import load_tif_stack, process_mask

# Load multi_slice_viewer (no .py extension → use SourceFileLoader directly)
import importlib.machinery
_loader = importlib.machinery.SourceFileLoader(
    "multi_slice_viewer", os.path.join(OWLO_DIR, "multi_slice_viewer")
)
_msv = _loader.load_module()
remove_keymap_conflicts = _msv.remove_keymap_conflicts

# ── pick volume ───────────────────────────────────────────────────────────────
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0

img_vol  = load_tif_stack(os.path.join(DATA_DIR, f"vol_linear_{k}.tif")).astype(np.float32) / 255.0
mask_vol = np.stack([process_mask(m)
                     for m in load_tif_stack(os.path.join(DATA_DIR, f"mask_{k}.tif"))])

print(f"Volume {k}  —  {img_vol.shape[0]} slices  {img_vol.shape[1]}×{img_vol.shape[2]} px")

# ── viewer ───────────────────────────────────────────────────────────────────
CMAP_MASK = mcolors.ListedColormap(["black", "deepskyblue", "tomato"])
remove_keymap_conflicts({"j", "k"})

fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Volume {k}  —  j = previous slice   k = next slice", fontsize=12)

mid = img_vol.shape[0] // 2
ax_img.index = ax_mask.index = mid

ax_img.imshow(img_vol[mid],  cmap="gray", vmin=0, vmax=1)
ax_img.set_title(f"Image  [{mid} / {img_vol.shape[0]}]")
ax_img.axis("off")

sc = ax_mask.imshow(mask_vol[mid], cmap=CMAP_MASK, vmin=0, vmax=2)
ax_mask.set_title(f"Mask   [{mid} / {mask_vol.shape[0]}]")
ax_mask.axis("off")
plt.colorbar(sc, ax=ax_mask, ticks=[0, 1, 2], label="0=bg  1=A  2=B")


def on_key(event):
    n = img_vol.shape[0]
    if event.key == "j":
        ax_img.index = (ax_img.index - 1) % n
    elif event.key == "k":
        ax_img.index = (ax_img.index + 1) % n
    else:
        return
    idx = ax_img.index
    ax_mask.index = idx
    ax_img.images[0].set_array(img_vol[idx])
    ax_mask.images[0].set_array(mask_vol[idx])
    ax_img.set_title(f"Image  [{idx} / {n}]")
    ax_mask.set_title(f"Mask   [{idx} / {n}]")
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.tight_layout()
plt.show()
