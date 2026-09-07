"""
FILE PURPOSE
    Detector adapter · Grounding DINO (IDEA Research).

WHAT IT IS
    Open-vocabulary transformer detector. You hand it plain-English phrases and it
    finds those things. No retraining to add an item type.

WHY IT'S THE BASELINE
    Accuracy leader among zero-shot detectors (52.5% AP on COCO with no COCO
    training data) and Apache 2.0, so it can ship. Slower than the YOLO family.

TRADE-OFF vs YOLO-World
    ~20x slower and ~5x larger, for a modest accuracy gain. On a 16-frame room
    that is seconds, not minutes, so accuracy wins for the POC.

LICENCE  Apache 2.0 — commercial use fine

HOW TO RUN
    Not an entry point. This adapter is selected by key:
        --detector grounding_dino
    passed to run_dataset_eval or run_combination.
    Confirm it is installed:  python -m poc.models.registry
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, pick_device, nms, normalise_label

INFO = ModelInfo(
    key="grounding_dino",
    kind="detector",
    display="Grounding DINO (base)",
    checkpoint="IDEA-Research/grounding-dino-base",
    licence="Apache-2.0",
    commercial_ok=True,
    open_vocabulary=True,
    gives_masks=False,
    pip_extra="transformers",
    notes="Zero-shot accuracy leader. Boxes only, no masks — pair with SAM 2 to get masks.",
)


class GroundingDinoDetector:
    info = INFO

    def __init__(self, box_threshold: float = 0.30, text_threshold: float = 0.25):
        self.box_th, self.txt_th = box_threshold, text_threshold
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._proc = AutoProcessor.from_pretrained(INFO.checkpoint)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                INFO.checkpoint).to(pick_device()).eval()
        return self._proc, self._model

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        import cv2, torch
        from PIL import Image
        proc, model = self._load()
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        # Grounding DINO expects lowercase phrases joined by ". " with a trailing period.
        text = ". ".join(p.lower() for p in prompts) + "."
        inp = proc(images=pil, text=text, return_tensors="pt").to(pick_device())
        with torch.no_grad():
            out = model(**inp)
        kw = dict(input_ids=inp["input_ids"], target_sizes=[pil.size[::-1]],
                  text_threshold=self.txt_th)
        # transformers renamed this argument around v4.51 — support both spellings.
        try:
            res = proc.post_process_grounded_object_detection(out, threshold=self.box_th, **kw)[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(out, box_threshold=self.box_th, **kw)[0]
        # "text_labels" FIRST, deliberately. transformers warns:
        #   "The key `labels` will return integer ids in
        #    post_process_grounded_object_detection output since v4.51.0.
        #    Use `text_labels` instead to retrieve string object names."
        # In 5.16.1 both keys still hold identical strings, so preferring
        # `labels` works today — and would break silently the moment that
        # change lands, because str(3) == "3" matches no prompt and EVERY
        # detection would arrive unnamed. Read the string key by name.
        labels = res.get("text_labels")
        if labels is None:                       # transformers < 4.51
            labels = res.get("labels") or []
        if labels and not isinstance(next(iter(labels)), str):
            raise TypeError(
                "Grounding DINO returned non-string labels "
                f"({type(next(iter(labels))).__name__}). transformers has changed "
                "its post-processing contract; map ids to prompts in this adapter "
                "before going further — do not let unnamed boxes through silently.")
        dets = []
        for l, sc, b in zip(labels, res["scores"], res["boxes"]):
            raw = str(l).strip().lower()
            # Normalise BEFORE nms, not after: nms groups by exact label, so raw
            # token spans would never suppress each other and the same object
            # would be counted several times. See normalise_label().
            lab, reason = normalise_label(raw, prompts)
            dets.append(Detection(label=lab, label_raw=raw, label_reason=reason,
                                  score=float(sc), box=[float(v) for v in b],
                                  source=INFO.key))
        return nms(dets)
