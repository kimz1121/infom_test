"""Shared frozen-ResNet34 image feature extractor (precompute multimodal inFOM).

THIS IS THE SINGLE SOURCE OF TRUTH for turning raw images into the visual half
of the multimodal observation. Both dataset construction
(`data_gen_scripts/extract_mm_features.py`) and inference/visualization MUST use
this exact module, so the 512-d features are bit-identical between train and
inference (verified deterministic: same input -> same output).

Transform (explicit, no center-crop so the full native frame is kept):
    uint8 (H,W,3) -> float/255 -> ImageNet mean/std normalize -> (3,H,W)
    -> torchvision resnet34(IMAGENET1K_V1) with fc removed -> 512-d feature.

The fused multimodal observation used by the (encoder=None) inFOM model is
    concat([resnet34_feature(512), proprio_state(D_state)])  ->  512 + D_state.
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESNET34_FEAT_DIM = 512


class FrozenResNet34Extractor:
    """Frozen ImageNet ResNet34 penultimate-feature extractor (eval, no grad).

    Weights are cached under ~/.cache/torch (bind-mounted), so no re-download.
    Determinism is guaranteed by eval mode + no dropout + fixed weights.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 256):
        import torch
        import torchvision

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet34(weights=weights)
        model.fc = torch.nn.Identity()  # -> 512-d penultimate feature
        self.model = model.eval().to(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def feat_dim(self) -> int:
        return RESNET34_FEAT_DIM

    def extract(self, images_uint8: np.ndarray) -> np.ndarray:
        """images_uint8: (N, H, W, 3) uint8 -> (N, 512) float32 features."""
        torch = self.torch
        assert images_uint8.ndim == 4 and images_uint8.shape[-1] == 3, images_uint8.shape
        out = []
        with torch.no_grad():
            for s in range(0, len(images_uint8), self.batch_size):
                chunk = images_uint8[s : s + self.batch_size]
                x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0  # (B,3,H,W) in [0,1]
                x = (x - self._mean) / self._std
                feat = self.model(x)  # (B, 512)
                out.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def fuse(self, images_uint8: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Return the fused (N, 512 + D_state) multimodal observation."""
        feat = self.extract(images_uint8)
        return np.concatenate([feat, np.asarray(state, dtype=np.float32)], axis=-1)


# ---------------------------------------------------------------------------
# DINOv3 patch-token extractor (learnable-attention-pool inFOM experiment).
#
# Unlike the ResNet path (one global 512-d vector/frame), DINOv3 gives a GRID of
# fine-grained patch tokens. We keep a small spatial grid (adaptive-pooled to
# POOL_GRID x POOL_GRID) plus the CLS token so a downstream *learnable* attention
# pooling can select relevant regions. The heavy frozen backbone runs ONCE here
# (precompute); training only learns the small attention pool over these tokens.
#
# Output per image: (1 + POOL_GRID*POOL_GRID, DINOV3_VITS16_DIM) tokens
#   token[0]         = CLS  (global)
#   token[1:]        = adaptive-pooled patch grid (row-major), fine-grained.
# Determinism: eval mode + no grad + fixed weights + deterministic adaptive pool.
# ---------------------------------------------------------------------------
DINOV3_VITS16_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_VITS16_DIM = 384
DINOV3_PATCH = 16
DINOV3_NUM_REGISTER = 4  # DINOv3 prepends 1 CLS + 4 register tokens before patches.


class FrozenDINOv3Extractor:
    """Frozen DINOv3 ViT-S/16 patch-token extractor (eval, no grad).

    Weights come from the gated HF repo (already access-granted, cached under
    ~/.cache/huggingface). Input images are fed at their native size (a multiple
    of 16, e.g. 256 -> 16x16=256 patch tokens); the patch grid is then
    adaptive-avg-pooled to ``pool_grid`` x ``pool_grid`` tokens.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 128,
                 pool_grid: int = 8, keep_cls: bool = True):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.pool_grid = pool_grid
        self.keep_cls = keep_cls
        model = AutoModel.from_pretrained(DINOV3_VITS16_NAME)
        self.model = model.eval().to(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def feat_dim(self) -> int:
        return DINOV3_VITS16_DIM

    @property
    def n_tokens(self) -> int:
        """Tokens emitted per image: pooled patch grid (+ CLS if kept)."""
        return self.pool_grid * self.pool_grid + (1 if self.keep_cls else 0)

    def extract(self, images_uint8: np.ndarray) -> np.ndarray:
        """images_uint8: (N,H,W,3) uint8 -> (N, n_tokens, 384) float32 tokens.

        H,W must be multiples of 16 (native, no resize). CLS is token 0 (if
        kept); the remaining tokens are the row-major adaptive-pooled patch grid.
        """
        torch = self.torch
        assert images_uint8.ndim == 4 and images_uint8.shape[-1] == 3, images_uint8.shape
        H, W = images_uint8.shape[1:3]
        assert H % DINOV3_PATCH == 0 and W % DINOV3_PATCH == 0, (
            f"H,W must be multiples of {DINOV3_PATCH}, got {(H, W)}")
        gh, gw = H // DINOV3_PATCH, W // DINOV3_PATCH
        n_skip = 1 + DINOV3_NUM_REGISTER  # CLS + register tokens precede patches
        out = []
        with torch.no_grad():
            for s in range(0, len(images_uint8), self.batch_size):
                chunk = images_uint8[s : s + self.batch_size]
                x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = (x - self._mean) / self._std
                lhs = self.model(x).last_hidden_state  # (B, 1+reg+gh*gw, 384)
                patches = lhs[:, n_skip:, :]           # (B, gh*gw, 384)
                B = patches.shape[0]
                grid = patches.reshape(B, gh, gw, -1).permute(0, 3, 1, 2)  # (B,384,gh,gw)
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    grid, (self.pool_grid, self.pool_grid))               # (B,384,pg,pg)
                pooled = pooled.flatten(2).permute(0, 2, 1)               # (B,pg*pg,384)
                if self.keep_cls:
                    cls = lhs[:, 0:1, :]                                  # (B,1,384)
                    toks = torch.cat([cls, pooled], dim=1)               # (B,1+pg*pg,384)
                else:
                    toks = pooled
                out.append(toks.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
