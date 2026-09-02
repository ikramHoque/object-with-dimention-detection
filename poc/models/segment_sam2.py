"""
FILE PURPOSE
    Segmenter adapter · SAM 2 (Meta). Upgrades boxes into masks.

WHY THIS EXISTS — AND WHY IT MAY MATTER MORE THAN THE DETECTOR CHOICE
    A bounding box around a sofa also contains the wall behind it and the floor in
    front. Method B measures the 3D spread of every point inside the box, so those
    background pixels get counted as part of the sofa and the object measures too
    large — sometimes much too large, for a sofa against a far wall.

    A mask keeps only the sofa's own pixels. This turns a systematic over-measurement
    into a much smaller one, for one extra model pass.

    So this adapter is a cheap way to give a box-only detector (Grounding DINO,
    OWLv2, RT-DETR) the same advantage SAM 3 and YOLOE-seg have natively. Whether
    that is worth the extra pass is exactly the kind of thing the sweep answers.

LICENCE  Apache 2.0 — commercial use fine
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import ModelInfo, pick_device

INFO = ModelInfo(
    key="sam2",
    kind="segmenter",
    display="SAM 2.1 Hiera-Large",
    checkpoint="facebook/sam2.1-hiera-large",
    licence="Apache-2.0",
    commercial_ok=True,
    gives_masks=True,
    pip_extra="transformers",
    notes="Box -> mask refinement. Removes background pixels from Method B's "
          "measurements, which is a systematic over-measurement otherwise.",
)


class Sam2Segmenter:
    info = INFO

    def __init__(self):
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, Sam2Model
            self._proc = AutoProcessor.from_pretrained(INFO.checkpoint)
            self._model = Sam2Model.from_pretrained(INFO.checkpoint).to(pick_device()).eval()
        return self._proc, self._model

    def refine(self, image_bgr: np.ndarray, boxes: Sequence[list[float]]) -> list[np.ndarray]:
        """One boolean mask per input box. Falls back to a filled box on failure."""
        import cv2, torch
        from PIL import Image
        H, W = image_bgr.shape[:2]
        if not boxes:
            return []
        try:
            proc, model = self._load()
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            inp = proc(images=pil, input_boxes=[[list(map(float, b)) for b in boxes]],
                       return_tensors="pt").to(pick_device())
            with torch.no_grad():
                out = model(**inp, multimask_output=False)
            masks = proc.post_process_masks(
                out.pred_masks, inp["original_sizes"])[0]
            arr = masks.squeeze(1).cpu().numpy().astype(bool)
            return [arr[i] for i in range(arr.shape[0])]
        except Exception as e:
            print(f"  ! SAM 2 refinement failed ({type(e).__name__}), using filled boxes")
            out = []
            for b in boxes:
                m = np.zeros((H, W), dtype=bool)
                x0, y0, x1, y1 = [int(max(0, v)) for v in b]
                m[y0:min(y1, H), x0:min(x1, W)] = True
                out.append(m)
            return out
