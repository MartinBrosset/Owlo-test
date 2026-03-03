"""
models.py
---------
Two segmentation model architectures:
  1. UNetResNet18  – UNet decoder with ResNet-18 encoder (pretrained on ImageNet)
  2. DINOv2Seg     – frozen DINOv2 ViT-S/14 backbone + lightweight decoder
Both output (B, num_classes, H, W) logits for num_classes=3:
  class 0 = background, class 1 = labels 1&2, class 2 = labels 3&4
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import Dinov2Model, SamModel, SegformerModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Bilinear upsample ×2, optional skip concatenation, then two ConvBNReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle possible size mismatch from odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 – UNet with ResNet-18 encoder
# ─────────────────────────────────────────────────────────────────────────────

class UNetResNet18(nn.Module):
    """
    UNet with a ResNet-18 encoder pretrained on ImageNet.

    Input:  (B, 1, H, W)  — single-channel grayscale images
    Output: (B, num_classes, H, W)  — logits (not softmax)

    Encoder feature map sizes (for 256×256 input):
        stem   → (B,  64, 128, 128)   stride 2
        pool   → (B,  64,  64,  64)   stride 4
        layer1 → (B,  64,  64,  64)   stride 4
        layer2 → (B, 128,  32,  32)   stride 8
        layer3 → (B, 256,  16,  16)   stride 16
        layer4 → (B, 512,   8,   8)   stride 32
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 1, pretrained: bool = True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # ── Encoder ─────────────────────────────────────────────────────────
        # Replace first conv: 3-ch → in_channels, keep same geometry
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        # Initialise from pretrained: average across input channels
        if pretrained:
            with torch.no_grad():
                self.stem[0].weight.copy_(
                    backbone.conv1.weight.mean(dim=1, keepdim=True)
                )

        self.pool    = backbone.maxpool   # /2
        self.layer1  = backbone.layer1    # 64-ch,  /4
        self.layer2  = backbone.layer2    # 128-ch, /8
        self.layer3  = backbone.layer3    # 256-ch, /16
        self.layer4  = backbone.layer4    # 512-ch, /32

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec4 = UpBlock(512, 256, 256)  # + skip from layer3
        self.dec3 = UpBlock(256, 128, 128)  # + skip from layer2
        self.dec2 = UpBlock(128,  64,  64)  # + skip from layer1
        self.dec1 = UpBlock( 64,  64,  64)  # + skip from stem
        self.dec0 = UpBlock( 64,   0,  32)  # no skip, up to full res

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Encoder
        e0 = self.stem(x)        # /2,  64-ch
        p  = self.pool(e0)       # /4
        e1 = self.layer1(p)      # /4,  64-ch
        e2 = self.layer2(e1)     # /8,  128-ch
        e3 = self.layer3(e2)     # /16, 256-ch
        e4 = self.layer4(e3)     # /32, 512-ch

        # Decoder
        d = self.dec4(e4, e3)    # /16
        d = self.dec3(d,  e2)    # /8
        d = self.dec2(d,  e1)    # /4
        d = self.dec1(d,  e0)    # /2
        d = self.dec0(d)         # /1

        out = self.head(d)
        # Ensure output matches input spatial size exactly
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 – DINOv2 ViT-S/14 + lightweight decoder
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Seg(nn.Module):
    """
    Segmentation model with a (optionally frozen) DINOv2 ViT-S/14 backbone.

    Input:  (B, 1, H, W)  — single-channel grayscale images
    Output: (B, num_classes, H, W)  — logits

    The image is resized to img_size×img_size internally (must be divisible by 14).
    Grayscale is repeated to 3 channels; ImageNet normalisation is applied.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,          # 224/14 = 16 patches per side
        freeze_backbone: bool = True,
    ):
        super().__init__()
        assert img_size % 14 == 0, "img_size must be divisible by 14 (DINOv2 patch size)"
        self.img_size   = img_size
        self.patch_size = 14
        self.grid       = img_size // 14  # number of patches per spatial dim

        # ImageNet normalisation buffers (not trainable params)
        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1))

        # ── Backbone ────────────────────────────────────────────────────────
        # Use HuggingFace transformers — compatible with Python 3.9
        # (torch.hub dinov2 uses X|Y syntax that requires Python 3.10+)
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        embed_dim = self.backbone.config.hidden_size  # 384 for ViT-S

        # ── Decoder ─────────────────────────────────────────────────────────
        # Progressive upsampling: grid→2×grid→4×grid→full res
        self.conv1 = ConvBNReLU(embed_dim, 256, kernel_size=1, padding=0)
        self.up1   = nn.Sequential(ConvBNReLU(256, 128), ConvBNReLU(128, 128))
        self.up2   = nn.Sequential(ConvBNReLU(128,  64), ConvBNReLU( 64,  64))
        self.head  = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # Resize & convert to 3-channel RGB
        x_r = F.interpolate(x, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)
        x_r = x_r.repeat(1, 3, 1, 1)           # (B, 3, img_size, img_size)

        # ImageNet normalisation
        x_r = (x_r - self.mean) / self.std

        # DINOv2 features via HuggingFace transformers
        # last_hidden_state: (B, 1 + grid*grid, embed_dim)  — index 0 is CLS token
        feats = self.backbone(pixel_values=x_r)
        patch_tokens = feats.last_hidden_state[:, 1:, :]   # drop CLS → (B, grid*grid, embed_dim)
        g = self.grid
        patch_tokens = patch_tokens.reshape(B, g, g, -1).permute(0, 3, 1, 2)
        # → (B, embed_dim, grid, grid)

        # Decode with progressive upsampling
        d = self.conv1(patch_tokens)                         # (B, 256, g, g)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up1(d)                                      # (B, 128, 2g, 2g)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up2(d)                                      # (B, 64, 4g, 4g)

        out = self.head(d)                                   # (B, num_classes, 4g, 4g)
        # Upsample to original input resolution
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 – SegFormer MiT-B2 encoder + UNet-style decoder
# ─────────────────────────────────────────────────────────────────────────────

class SegFormerSeg(nn.Module):
    """
    SegFormer MiT-B2 encoder (pretrained, optionally frozen) + UNet decoder.

    Input:  (B, 1, H, W)  — single-channel grayscale (any size divisible by 4)
    Output: (B, num_classes, H, W)  — logits

    MiT-B2 produces 4 multi-scale feature maps (for H×W input):
        stage 0: (B,  64, H/4,  W/4)
        stage 1: (B, 128, H/8,  W/8)
        stage 2: (B, 320, H/16, W/16)
        stage 3: (B, 512, H/32, W/32)

    The UNet decoder fuses all 4 scales via skip connections, giving much
    better spatial detail than single-scale models (DINOv2, SAM).

    Trainable params (decoder only, encoder frozen):
        variant='b1' → ~2.1 M  (encoder ~55 MB)   ← default, good for CPU
        variant='b2' → ~2.1 M  (encoder ~100 MB)  ← more capacity, same decoder
    Channel dims are identical for b1–b5 so the decoder is reusable across variants.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    # Channel widths per MiT variant (b1–b5 share the same dims; b0 is smaller)
    _CHANNELS = {
        "b0": (32,  64,  160, 256),
        "b1": (64,  128, 320, 512),
        "b2": (64,  128, 320, 512),
        "b3": (64,  128, 320, 512),
        "b4": (64,  128, 320, 512),
        "b5": (64,  128, 320, 512),
    }

    def __init__(self, num_classes: int = 3, freeze_encoder: bool = True,
                 variant: str = "b1"):
        super().__init__()
        assert variant in self._CHANNELS, f"variant must be one of {list(self._CHANNELS)}"

        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1))

        self.backbone = SegformerModel.from_pretrained(
            f"nvidia/mit-{variant}", low_cpu_mem_usage=True,
        )
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        c0, c1, c2, c3 = self._CHANNELS[variant]

        # UNet-style decoder: bottleneck → progressive upsampling with skip connections
        self.conv3     = ConvBNReLU(c3, 256, kernel_size=1, padding=0)  # reduce bottleneck
        self.dec2      = UpBlock(256, c2, 256)   # skip from stage 2  (H/16)
        self.dec1      = UpBlock(256, c1, 128)   # skip from stage 1  (H/8)
        self.dec0      = UpBlock(128, c0,  64)   # skip from stage 0  (H/4)
        self.dec_final = UpBlock( 64,  0,  32)   # no skip, upsample to H/2
        self.head      = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # Grayscale → 3-channel + ImageNet normalisation
        x_r = x.expand(-1, 3, -1, -1).contiguous()
        x_r = (x_r - self.mean) / self.std

        # SegFormer encoder → 4 multi-scale feature maps (B, C, h, w)
        enc_out = self.backbone(pixel_values=x_r, output_hidden_states=True)
        s0, s1, s2, s3 = enc_out.hidden_states   # (B,64,H/4), (B,128,H/8), (B,320,H/16), (B,512,H/32)

        # UNet decoder
        d = self.conv3(s3)     # (B, 256, H/32, W/32)
        d = self.dec2(d, s2)   # (B, 256, H/16, W/16)
        d = self.dec1(d, s1)   # (B, 128, H/8,  W/8)
        d = self.dec0(d, s0)   # (B,  64, H/4,  W/4)
        d = self.dec_final(d)  # (B,  32, H/2,  W/2)
        out = self.head(d)     # (B, num_classes, H/2, W/2)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 – Lightweight 3D UNet
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock3D(nn.Sequential):
    """Two 3D conv-BN-ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet3D(nn.Module):
    """
    Lightweight 3D UNet for center-slice segmentation.

    Input:  (B, 1, D, H, W)  — D consecutive slices (D must be odd, default 5)
    Output: (B, num_classes, H, W)  — segmentation of the center slice only

    Strategy
    --------
    - 3D encoder captures context across all D slices.
    - Spatial pooling (H, W) at every level; depth pooled only once at the
      deepest level to keep D alive for as long as possible.
    - At the bottleneck the center slice is extracted, and the decoder
      runs purely in 2D using center-slice skip features.

    Channel widths: base_ch × [1, 2, 4, 8]  (default 16 → 16,32,64,128)
    Keeps the model light enough for CPU / small datasets.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 1,
                 depth: int = 5, base_ch: int = 16):
        super().__init__()
        assert depth % 2 == 1, "depth must be odd so a unique center slice exists"
        c = base_ch

        # ── 3D Encoder ──────────────────────────────────────────────────────
        self.enc1 = ConvBlock3D(in_channels, c)       # (B, c,  D,   H,   W)
        self.enc2 = ConvBlock3D(c,    c * 2)          # (B, 2c, D,   H/2, W/2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)         # (B, 4c, D,   H/4, W/4)
        self.enc4 = ConvBlock3D(c * 4, c * 8)         # (B, 8c, D/2, H/8, W/8)

        # Pool only in (H,W) to preserve depth as long as possible
        self.pool_xy  = nn.MaxPool3d(kernel_size=(1, 2, 2))
        # Pool in all dims once (bottleneck transition)
        self.pool_xyz = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # ── 2D Decoder (uses center-slice skip features) ─────────────────────
        self.dec4 = UpBlock(c * 8, c * 4, c * 4)
        self.dec3 = UpBlock(c * 4, c * 2, c * 2)
        self.dec2 = UpBlock(c * 2, c,     c)
        self.dec1 = UpBlock(c,     0,     c // 2)     # no skip at finest level

        self.head = nn.Conv2d(c // 2, num_classes, kernel_size=1)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> (B, C, H, W): extract center slice along D."""
        return x[:, :, x.shape[2] // 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]

        # ── Encode ──────────────────────────────────────────────────────────
        e1 = self.enc1(x)                          # (B, c,  D,   H,   W)
        e2 = self.enc2(self.pool_xy(e1))           # (B, 2c, D,   H/2, W/2)
        e3 = self.enc3(self.pool_xy(e2))           # (B, 4c, D,   H/4, W/4)
        e4 = self.enc4(self.pool_xyz(e3))          # (B, 8c, D/2, H/8, W/8)

        # ── Collapse depth → 2D ─────────────────────────────────────────────
        b  = self._center(e4)                      # (B, 8c, H/8, W/8)
        s3 = self._center(e3)                      # (B, 4c, H/4, W/4)
        s2 = self._center(e2)                      # (B, 2c, H/2, W/2)
        s1 = self._center(e1)                      # (B, c,  H,   W)

        # ── Decode ──────────────────────────────────────────────────────────
        d = self.dec4(b,  s3)                      # (B, 4c, H/4, W/4)
        d = self.dec3(d,  s2)                      # (B, 2c, H/2, W/2)
        d = self.dec2(d,  s1)                      # (B, c,  H,   W)
        d = self.dec1(d)                           # (B, c/2, 2H, 2W)

        out = self.head(d)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 – SegFormer MiT-B1 encoder + depth fusion + UNet decoder  (3-D input)
# ─────────────────────────────────────────────────────────────────────────────

class DepthFusion(nn.Module):
    """
    Learnable soft-attention weighted sum over D per-slice feature maps.

    Input:  (B, C, D, h, w)
    Output: (B, C, h, w)

    Initialised so that the centre slice dominates (logit=4, others=0).
    During training the weights are free to incorporate context from
    neighbouring slices.
    """

    def __init__(self, depth: int):
        super().__init__()
        init = torch.zeros(depth)
        init[depth // 2] = 4.0           # large logit → centre slice ~98 % weight initially
        self.weight = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weight, dim=0)           # (D,)
        return (x * w[None, None, :, None, None]).sum(dim=2)   # (B, C, h, w)


class SegFormerSeg3D(nn.Module):
    """
    SegFormer MiT-B1/B2 encoder (pretrained, optionally frozen) +
    learnable depth fusion + UNet decoder — for 3-D patch input.

    Input:  (B, 1, D, H, W)  — D consecutive grayscale slices (same format as UNet3D)
    Output: (B, num_classes, H, W)  — segmentation of the centre slice

    Strategy
    --------
    1. All D slices are encoded in a single batched MiT pass: (B*D, 1, H, W).
    2. At each of the 4 encoder scales a DepthFusion layer produces one
       centre-aware feature map from the D per-slice maps.
       (initialised to the centre slice; learns to use context from neighbours)
    3. The 4 fused feature maps feed the same UNet decoder as SegFormerSeg.

    Trainable parameters (encoder frozen):
        decoder     ~2.1 M   (same as SegFormerSeg)
        depth fuse    4 × D  (e.g. 4 × 5 = 20 weights — negligible)
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    _CHANNELS = {
        "b0": (32,  64,  160, 256),
        "b1": (64,  128, 320, 512),
        "b2": (64,  128, 320, 512),
        "b3": (64,  128, 320, 512),
        "b4": (64,  128, 320, 512),
        "b5": (64,  128, 320, 512),
    }

    def __init__(self, num_classes: int = 3, freeze_encoder: bool = True,
                 variant: str = "b1", depth: int = 5):
        super().__init__()
        assert variant in self._CHANNELS, f"variant must be one of {list(self._CHANNELS)}"
        self.depth   = depth
        self.variant = variant

        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1))

        self.backbone = SegformerModel.from_pretrained(
            f"nvidia/mit-{variant}", low_cpu_mem_usage=True,
        )
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        c0, c1, c2, c3 = self._CHANNELS[variant]

        # One depth-fusion layer per encoder scale (tiny — just D learnable weights each)
        self.fuse0 = DepthFusion(depth)
        self.fuse1 = DepthFusion(depth)
        self.fuse2 = DepthFusion(depth)
        self.fuse3 = DepthFusion(depth)

        # UNet decoder — identical to SegFormerSeg
        self.conv3     = ConvBNReLU(c3, 256, kernel_size=1, padding=0)
        self.dec2      = UpBlock(256, c2, 256)
        self.dec1      = UpBlock(256, c1, 128)
        self.dec0      = UpBlock(128, c0,  64)
        self.dec_final = UpBlock( 64,  0,  32)
        self.head      = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape
        c0, c1, c2, c3 = self._CHANNELS[self.variant]

        # ── Encode all D slices in one batched pass ───────────────────────────
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
        x_r    = x_flat.expand(-1, 3, -1, -1).contiguous()
        x_r    = (x_r - self.mean) / self.std

        enc_out = self.backbone(pixel_values=x_r, output_hidden_states=True)
        s0_flat, s1_flat, s2_flat, s3_flat = enc_out.hidden_states  # each (B*D, C, h, w)

        # ── Reshape to (B, C, D, h, w) and fuse across depth ─────────────────
        def to_depth(s, ch):
            h, w = s.shape[-2], s.shape[-1]
            return s.reshape(B, D, ch, h, w).permute(0, 2, 1, 3, 4)

        s0 = self.fuse0(to_depth(s0_flat, c0))   # (B, c0, H/4,  W/4)
        s1 = self.fuse1(to_depth(s1_flat, c1))   # (B, c1, H/8,  W/8)
        s2 = self.fuse2(to_depth(s2_flat, c2))   # (B, c2, H/16, W/16)
        s3 = self.fuse3(to_depth(s3_flat, c3))   # (B, c3, H/32, W/32)

        # ── UNet decoder ──────────────────────────────────────────────────────
        d = self.conv3(s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        d = self.dec0(d, s0)
        d = self.dec_final(d)
        out = self.head(d)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model 6 – SAM ViT-B encoder (frozen) + 2D semantic decoder
# ─────────────────────────────────────────────────────────────────────────────

class SAMSeg2D(nn.Module):
    """
    SAM ViT-B image encoder (frozen) + lightweight 2D semantic decoder.

    SAM requires EXACTLY 1024×1024 input (hard check in SamPatchEmbeddings).
    Running the encoder each training step on CPU (~5s/image) is impractical.
    Use the pre-extraction workflow in train_SAM.ipynb instead:
      1. Extract all embeddings once  (slow, run once)
      2. Train only the decoder       (fast, run many times)

    forward() auto-detects input by channel dimension:
      (B, 1,   H, W)        raw image        → encodes internally (inference only)
      (B, 256, 64, 64)      pre-cached emb.  → skips encoder     (training)

    SAM at 1024×1024 → (B, 256, 64, 64) features.
    Decoder: 64×64 → 128×128 → 256×256 → resize to target_size.
    ~460 K trainable parameters (decoder only).
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, num_classes: int = 3, freeze_encoder: bool = True,
                 target_size: tuple = (128, 128)):
        super().__init__()
        self.target_size = target_size

        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.encoder = sam.vision_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        embed_dim = 256   # SAM ViT-B at 1024×1024 → (B, 256, 64, 64)

        # Decoder: 2 × ×2: 64×64 → 128×128 → 256×256 → resize
        self.up1 = nn.Sequential(ConvBNReLU(embed_dim, 128), ConvBNReLU(128, 128))
        self.up2 = nn.Sequential(ConvBNReLU(128, 64),        ConvBNReLU(64,  64))
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1))

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) raw image → (B, 256, 64, 64) SAM embedding at 1024×1024."""
        x_r = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x_r = x_r.expand(-1, 3, -1, -1).contiguous()
        x_r = (x_r - self.mean) / self.std
        return self.encoder(pixel_values=x_r).last_hidden_state

    def _decode(self, features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        d = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up1(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up2(d)
        out = self.head(d)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            # Raw image (B, 1, H, W) — used at inference time
            H, W = x.shape[-2], x.shape[-1]
            features = self._extract(x)
        else:
            # Pre-extracted embedding (B, 256, 64, 64) — used during training
            features = x
            H, W = self.target_size
        return self._decode(features, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Model 6 – SAM ViT-B encoder (frozen) + 3D-aware semantic decoder
# ─────────────────────────────────────────────────────────────────────────────

class SAMSeg3D(nn.Module):
    """
    SAM ViT-B image encoder (frozen) applied per-slice, features averaged
    across depth for 3-D context, decoded to center-slice segmentation.

    forward() auto-detects input by shape:
      (B, 1,  D, H, W)       raw 3-D patch        → encodes D slices (slow)
      (B, D, 256, 64, 64)    pre-cached per-slice  → averages + decodes (fast)

    ~460 K trainable parameters (same decoder as SAMSeg2D).
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, num_classes: int = 3, freeze_encoder: bool = True,
                 target_size: tuple = (128, 128)):
        super().__init__()
        self.target_size = target_size

        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.encoder = sam.vision_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        embed_dim = 256

        self.up1 = nn.Sequential(ConvBNReLU(embed_dim, 128), ConvBNReLU(128, 128))
        self.up2 = nn.Sequential(ConvBNReLU(128, 64),        ConvBNReLU(64,  64))
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1))

    def _decode(self, features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        d = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up1(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up2(d)
        out = self.head(d)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            # Raw 3-D patch (B, 1, D, H, W) — inference only
            B, C, D, H, W = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            x_flat = F.interpolate(x_flat, size=(1024, 1024),
                                    mode="bilinear", align_corners=False)
            x_flat = x_flat.expand(-1, 3, -1, -1).contiguous()
            x_flat = (x_flat - self.mean) / self.std
            feats_flat = self.encoder(pixel_values=x_flat).last_hidden_state
            S = feats_flat.shape[-1]
            features = feats_flat.reshape(B, D, 256, S, S).mean(dim=1)
        else:
            # Pre-extracted per-slice embeddings (B, D, 256, 64, 64) — training
            features = x.mean(dim=1)        # average across depth → (B, 256, 64, 64)
            H, W = self.target_size
        return self._decode(features, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 256, 256).to(device)

    print("── UNetResNet18 ──")
    m1 = UNetResNet18(num_classes=3).to(device)
    y1 = m1(x)
    print(f"  input {tuple(x.shape)} → output {tuple(y1.shape)}")
    n_params = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    print("\n── DINOv2Seg ──")
    m2 = DINOv2Seg(num_classes=3, img_size=224, freeze_backbone=True).to(device)
    y2 = m2(x)
    print(f"  input {tuple(x.shape)} → output {tuple(y2.shape)}")
    n_params = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    print("\n── UNet3D ──")
    x3 = torch.randn(2, 1, 5, 128, 128).to(device)
    m3 = UNet3D(num_classes=3, depth=5, base_ch=16).to(device)
    y3 = m3(x3)
    print(f"  input {tuple(x3.shape)} → output {tuple(y3.shape)}")
    n_params = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    print("\n── SegFormerSeg3D ──")
    x4 = torch.randn(2, 1, 5, 128, 128).to(device)
    m4 = SegFormerSeg3D(num_classes=3, variant="b1", depth=5).to(device)
    y4 = m4(x4)
    print(f"  input {tuple(x4.shape)} → output {tuple(y4.shape)}")
    n_params = sum(p.numel() for p in m4.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")
