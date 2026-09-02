"""
FILE PURPOSE
    Depth adapter · UniDepthV2 (ETH Zurich). RESEARCH COMPARISON ONLY.

WHAT IT DOES
    ViT encoder plus a self-promptable camera module and a pseudo-spherical output
    that separates ray direction from log-depth. Strong indoor numbers: delta1 96.4
    on SUN-RGBD, 94.5 on iBims-1, 90.5 on TUM-RGBD, and 25ms inference.

##############################################################################
#  THIS MODEL CANNOT SHIP. IT IS HERE TO ESTABLISH A CEILING, NOTHING ELSE.
#
#  UniDepthV2 is CC BY-NC 4.0 — non-commercial. It posts some of the best indoor
#  metric depth numbers in the field and NX could never use it in a product.
#
#  Its only legitimate role in this project: tell us how much accuracy we are
#  giving up by being licence-clean. If MoGe-2 lands close to it, the licence
#  costs us nothing and the question is settled. If the gap is large, that is an
#  argument for either a commercial licence negotiation or hardware depth.
#
#  Registered commercial_ok=False. The runner refuses it without
#  --allow-noncommercial, and results are tagged so they can never be mistaken
#  for a shippable configuration.
##############################################################################
"""
from __future__ import annotations
import numpy as np
from .base import DepthResult, ModelInfo, pick_device

INFO = ModelInfo(
    key="unidepth2",
    kind="depth",
    display="UniDepthV2 ViT-L",
    checkpoint="lpiccinelli/unidepth-v2-vitl14",
    licence="CC BY-NC 4.0",
    commercial_ok=False,
    pip_extra="git+https://github.com/lpiccinelli-eth/UniDepth.git",
    notes="Best-in-class indoor metric depth, NON-COMMERCIAL. Use only to measure the "
          "accuracy cost of staying licence-clean.",
    warn="CC BY-NC 4.0 — CANNOT SHIP. Research ceiling measurement only.",
)


class UniDepthV2Depth:
    info = INFO

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from unidepth.models import UniDepthV2
            self._model = UniDepthV2.from_pretrained(INFO.checkpoint).to(pick_device()).eval()
        return self._model

    def infer(self, image_bgr: np.ndarray) -> DepthResult:
        import cv2, torch
        model = self._load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)      # UniDepth wants uint8 CHW
        with torch.no_grad():
            pred = model.infer(t.to(pick_device()))
        depth = pred["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        pts = pred["points"].squeeze().detach().cpu().numpy().astype(np.float32)
        if pts.shape[0] == 3:                            # (3,H,W) -> (H,W,3)
            pts = np.transpose(pts, (1, 2, 0))
        K = pred.get("intrinsics")
        if K is not None:
            K = K.squeeze().detach().cpu().numpy()
        return DepthResult(points=pts, depth=depth,
                           mask=np.isfinite(depth) & (depth > 0),
                           intrinsics=K, vert_axis=1, source=INFO.key)
