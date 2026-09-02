"""
FILE PURPOSE
    Detector adapter · SAM 3 (Meta, November 2025).

WHAT IT IS
    Promptable concept segmentation. Give it a concept and it finds and SEGMENTS every
    instance in one pass. Trained on 4 million concepts, supports ~270,000 at inference.

WHY IT'S THE MOST PROMISING OPTION
    It returns MASKS, not just boxes. That matters more for us than headline accuracy:
    a box around a sofa contains wall and floor pixels, so Method B measures the box's
    contents rather than the sofa and over-reads. A mask removes that error for free.

ON THE BENCHMARK NUMBERS
    SAM 3 scores cgF1 54.1 on SA-Co Gold against Grounding DINO's 3.3. Do not read
    that as "16x better at detection" — SA-Co is Meta's own benchmark, built around
    SAM 3's task formulation, and Grounding DINO is penalised partly for format
    mismatch. SAM 3 is clearly better at open-vocabulary instance segmentation; the
    margin is not 16x.

SETUP  Weights are gated on Hugging Face — accept the licence and `huggingface-cli login`.

LICENCE  Meta custom licence. Commercial use permitted; restrictions cover military,
         ITAR, nuclear and weapons applications, none of which apply to removals.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, pick_device, nms

INFO = ModelInfo(
    key="sam3",
    kind="detector",
    display="SAM 3 (concept segmentation)",
    checkpoint="facebook/sam3",
    licence="Meta SAM licence",
    commercial_ok=True,
    open_vocabulary=True,
    gives_masks=True,
    pip_extra="transformers",
    notes="Returns masks. Gated weights — accept the licence and log in to HF first.",
    warn="Gated checkpoint. If loading fails, run: huggingface-cli login",
)


class Sam3Detector:
    info = INFO

    def __init__(self, score_threshold: float = 0.35):
        self.th = score_threshold
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            # API surface is new and still settling; try the dedicated classes first.
            from transformers import AutoProcessor
            self._proc = AutoProcessor.from_pretrained(INFO.checkpoint)
            try:
                from transformers import Sam3Model
                self._model = Sam3Model.from_pretrained(INFO.checkpoint)
            except ImportError:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(INFO.checkpoint)
            self._model = self._model.to(pick_device()).eval()
        return self._proc, self._model

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        import cv2, torch
        from PIL import Image
        proc, model = self._load()
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        dets: list[Detection] = []
        # SAM 3 takes one concept at a time and returns every instance of it.
        for phrase in prompts:
            inp = proc(images=pil, text=phrase.lower(), return_tensors="pt").to(pick_device())
            with torch.no_grad():
                out = model(**inp)
            res = proc.post_process_instance_segmentation(
                out, threshold=self.th, target_sizes=[pil.size[::-1]])[0]
            boxes = res.get("boxes", [])
            scores = res.get("scores", [])
            masks = res.get("masks", [None] * len(boxes))
            for b, s, m in zip(boxes, scores, masks):
                mask = None
                if m is not None:
                    mask = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
                dets.append(Detection(label=phrase.lower(), score=float(s),
                                      box=[float(v) for v in b], mask=mask, source=INFO.key))
        return nms(dets)
