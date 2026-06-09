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
