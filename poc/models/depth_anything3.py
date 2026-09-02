"""
FILE PURPOSE
    Depth adapter · Depth Anything 3 (ByteDance, Nov 2025) — metric variant.

WHAT IT DOES
    One plain Vision Transformer that handles single OR multiple views, alternating
    within-view and cross-view attention. State of the art on ETH3D (delta1 0.917,
    AbsRel 0.104) and reported 44.3% better camera pose / 25.1% better geometry than
    VGGT.

WHY IT'S WORTH A COMPARISON RUN
    The multi-view capability is the interesting part for us. A guided video pan gives
    genuine multi-view constraints that a single image cannot, and DA3 can exploit
    them without a separate structure-from-motion stage. If it beats MoGe-2 on our
    footage, the capture instruction ("slow pan") starts paying for itself twice.

##############################################################################
#  LICENCE TRAP — the variant matters, not the project
#
#  DA3-GIANT, DA3-LARGE and the NESTED variants are CC BY-NC 4.0 — NON-COMMERCIAL.
#  DA3METRIC-LARGE, DA3MONO-LARGE, DA3-BASE and DA3-SMALL are Apache 2.0.
#
#  This adapter defaults to DA3METRIC-LARGE precisely because it is the Apache one
#  AND the metric one. Do not casually switch to DA3-LARGE for "better numbers" —
#  it cannot ship.
##############################################################################
"""
from __future__ import annotations
import numpy as np
from .base import DepthResult, ModelInfo, pick_device

INFO = ModelInfo(
    key="da3_metric",
    kind="depth",
    display="Depth Anything 3 Metric-Large",
    checkpoint="depth-anything/DA3METRIC-LARGE",
    licence="Apache-2.0",
    commercial_ok=True,
    pip_extra="transformers",
    notes="Apache-licensed metric variant. Handles single and multi-view. "
          "DA3-LARGE/GIANT/NESTED are CC BY-NC — do not substitute them.",
    warn="Checkpoint naming and the HF integration for DA3 are new; if loading fails, "
         "check the model card for the current id and API.",
)


class DepthAnything3Depth:
    info = INFO

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint or INFO.checkpoint
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._proc = AutoImageProcessor.from_pretrained(self.checkpoint)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self.checkpoint).to(pick_device()).eval()
        return self._proc, self._model

    def infer(self, image_bgr: np.ndarray) -> DepthResult:
        import cv2, torch
        from PIL import Image
        proc, model = self._load()
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=pil, return_tensors="pt").to(pick_device())
        with torch.no_grad():
            out = model(**inp)
        depth = proc.post_process_depth_estimation(
            out, target_sizes=[pil.size[::-1]])[0]["predicted_depth"]
        depth = depth.detach().cpu().numpy().astype(np.float32)

        # DA3 returns a depth map, not a point map. Back-project to 3D ourselves using
        # the intrinsics the processor exposes, or a sane default field of view.
        H, W = depth.shape
        fx = fy = float(getattr(model.config, "focal_length", 0) or 0) or 0.7 * W
        cx, cy = W / 2.0, H / 2.0
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        X = (xs - cx) / fx * depth
        Y = (ys - cy) / fy * depth
        points = np.stack([X, Y, depth], axis=-1)
        mask = np.isfinite(depth) & (depth > 0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return DepthResult(points=points, depth=depth, mask=mask,
                           intrinsics=K, vert_axis=1, source=INFO.key)
