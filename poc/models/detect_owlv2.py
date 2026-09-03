"""
FILE PURPOSE
    Detector adapter · OWLv2 (Google).

WHAT IT IS
    Open-vocabulary detector trained with the OWL-ST self-training recipe on over a
    billion pseudo-annotated examples.

WHY INCLUDE IT
    It was built specifically to improve on RARE and LONG-TAIL categories. A removal
    survey is full of long tail — dressing tables, rolled rugs, wardrobe cartons — so
    it may beat Grounding DINO on exactly the items that matter to us, even though it
    loses on common-object benchmarks.

LICENCE  Apache 2.0 — commercial use fine

HOW TO RUN
    Not an entry point. This adapter is selected by key:
        --detector owlv2
    passed to run_dataset_eval or run_combination.
    Confirm it is installed:  python -m poc.models.registry
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, pick_device, nms

INFO = ModelInfo(
    key="owlv2",
    kind="detector",
    display="OWLv2 (base, ensemble)",
    checkpoint="google/owlv2-base-patch16-ensemble",
    licence="Apache-2.0",
    commercial_ok=True,
    open_vocabulary=True,
    gives_masks=False,
    pip_extra="transformers",
    notes="Built for rare/long-tail categories. Worth testing precisely because our "
          "vocabulary is unusual.",
)


class Owlv2Detector:
    info = INFO

    def __init__(self, box_threshold: float = 0.20):
        self.box_th = box_threshold
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            self._proc = Owlv2Processor.from_pretrained(INFO.checkpoint)
            self._model = Owlv2ForObjectDetection.from_pretrained(
                INFO.checkpoint).to(pick_device()).eval()
        return self._proc, self._model

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        import cv2, torch
        from PIL import Image
        proc, model = self._load()
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        # OWLv2 takes a nested list of queries and prefers natural phrasing.
        queries = [[f"a photo of a {p}" for p in prompts]]
        inp = proc(text=queries, images=pil, return_tensors="pt").to(pick_device())
        with torch.no_grad():
            out = model(**inp)
        target = torch.tensor([[pil.size[1], pil.size[0]]], device=pick_device())
        try:
            res = proc.post_process_grounded_object_detection(
                outputs=out, target_sizes=target, threshold=self.box_th)[0]
        except (AttributeError, TypeError):
            res = proc.post_process_object_detection(
                outputs=out, target_sizes=target, threshold=self.box_th)[0]
        dets = []
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            idx = int(l)
            if idx >= len(prompts):
                continue
            dets.append(Detection(label=prompts[idx].lower(), score=float(s),
                                  box=[float(v) for v in b], source=INFO.key))
        return nms(dets)
