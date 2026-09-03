"""
FILE PURPOSE
    Depth adapter · MoGe-2 (Microsoft, NeurIPS 2025). The default and the safe choice.

WHAT IT DOES
    Predicts a metric-scale 3D point map from a single image — every pixel gets an
    (X, Y, Z) in metres — plus camera intrinsics and a confidence mask.

WHY IT'S THE DEFAULT
    Best published metric point-map error among models we can actually ship: 8.19%
    relative error, 93.6% inliers. UniDepthV2 does better on some indoor benchmarks
    but is CC BY-NC and can never ship, so it is comparison-only here.

WHAT THAT ERROR MEANS FOR US
    8.19% on length becomes roughly 26% on volume, because volume goes as length
    cubed. And it is ONE error applied to the whole room, so it does not average out
    across objects. That is the entire reason the scale-anchor stage exists.

LICENCE  MIT (DINOv2 components Apache 2.0) — commercial use fine

HOW TO RUN
    Not an entry point. This adapter is selected by key:
        --depth moge2
    passed to run_combination.
    Confirm it is installed:  python -m poc.models.registry
"""
from __future__ import annotations
import numpy as np
from .base import DepthResult, ModelInfo, pick_device

INFO = ModelInfo(
    key="moge2",
    kind="depth",
    display="MoGe-2 ViT-L (metric)",
    checkpoint="Ruicheng/moge-2-vitl",
    licence="MIT",
    commercial_ok=True,
    pip_extra="git+https://github.com/microsoft/MoGe.git",
    notes="8.19% metric point-map error, 326M params, ~29ms/frame FP16. "
          "Smaller variants: moge-2-vitb-normal, moge-2-vits-normal.",
)

# Smaller checkpoints, for a slow machine.
VARIANTS = {"vitl": "Ruicheng/moge-2-vitl",
            "vitb": "Ruicheng/moge-2-vitb-normal",
            "vits": "Ruicheng/moge-2-vits-normal"}


class Moge2Depth:
    info = INFO

    def __init__(self, variant: str = "vitl"):
        self.checkpoint = VARIANTS.get(variant, INFO.checkpoint)
        self._model = None

    def _load(self):
        if self._model is None:
            from moge.model.v2 import MoGeModel
            self._model = MoGeModel.from_pretrained(self.checkpoint).to(pick_device()).eval()
        return self._model

    def infer(self, image_bgr: np.ndarray) -> DepthResult:
        import cv2, torch
        model = self._load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=pick_device()).permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(t)
        g = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in o.items()}
        return DepthResult(points=g["points"], depth=g["depth"],
                           mask=g["mask"].astype(bool),
                           intrinsics=g.get("intrinsics"),
                           vert_axis=1, source=INFO.key)
