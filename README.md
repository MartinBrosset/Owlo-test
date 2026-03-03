# Owlo-test — Cell Segmentation Pipeline

Semantic segmentation of 2-cell mouse embryo fluorescence microscopy volumes.
The pipeline trains and evaluates multiple deep learning models on synthetic data, then tries to applies them zero-shot to real confocal microscopy data.

---


## Project Structure

```
Owlo-test/
│
├── utils/                          # Shared Python modules
│   ├── models.py                   # All model architectures (7 models)
│   ├── train_functions.py          # Datasets, losses, training loop, helpers
│   └── view_volume.py              # Interactive volume viewer
│
├── model_weights/                  # Saved checkpoints (.pth)
│   ├── best_unet_resnet18.pth
│   ├── best_dinov2_seg.pth
│   ├── best_segformer.pth
│   ├── best_segformer_ft.pth       # Fine-tuned SegFormer (optional)
│   ├── best_unet3d.pth
│   ├── best_segformer3d.pth
│   ├── best_samseg2d.pth           # SAM-based (experimental)
│   └── best_samseg3d.pth
│
├── notebooks/
│   ├── training/                   # One notebook per model (or model pair)
│   │   ├── train_UNet_Dinov2Seg.ipynb   # UNetResNet18 + DINOv2Seg
│   │   ├── train_SegFormer.ipynb        # SegFormerSeg (2D)
│   │   ├── train_3D.ipynb              # UNet3D + SegFormerSeg3D
│   │   └── train_SAM.ipynb             # SAMSeg2D + SAMSeg3D (experimental)
│   │
│   └── visualisation/              # Analysis & exploration
│       ├── evaluate.ipynb          # ← START HERE to compare all models
│       ├── distinguish_cells.ipynb # Instance separation (4 labels from 3 classes)
│       └── explore_real_data.ipynb # Zero-shot transfer to real mouse embryo data
│
├── sam_embeddings/                 # Pre-extracted SAM encoder features (generated)
│   └── emb_{k}.npz
│
├── RAG-project/                    # Python virtual environment (.venv)
├── .gitignore
└── README.md
```
