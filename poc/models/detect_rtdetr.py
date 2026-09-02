"""
FILE PURPOSE
    Detector adapter · RT-DETR (Baidu). The licence-clean speed option.

WHAT IT IS
    Real-time detection transformer. Apache 2.0, so unlike the Ultralytics YOLO
    family it can ship in a closed-source product with no obligations.

THE TRADE
    It is CLOSED-vocabulary — it only knows the classes it was trained on. This
    checkpoint is trained on Objects365 (365 classes) then COCO, which covers far
    more household objects than COCO's 80 alone, but it still cannot be told to look
    for something new. Adding an item type means retraining, not editing a JSON file.

WHY INCLUDE IT ANYWAY
    It answers a specific question: how much accuracy do we actually lose by giving
    up open vocabulary? If a fixed 365-class model covers 90% of a removal survey,
    the speed and licence clarity may be worth more than open vocabulary.

VOCABULARY GAP TO WATCH
    Our vocabulary contains items no COCO-style dataset has: wardrobe, chest of
    drawers, sideboard, dressing table, mattress, washing machine, dishwasher,
    cardboard box, rolled rug. Those will simply be missed. The runner records which
    prompts a closed-vocab model could not even attempt, so the comparison stays fair.

LICENCE  Apache 2.0 — commercial use fine
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, pick_device, nms

INFO = ModelInfo(
    key="rtdetr",
    kind="detector",
    display="RT-DETR R50 (Objects365 + COCO)",
    checkpoint="PekingU/rtdetr_r50vd_coco_o365",
    licence="Apache-2.0",
    commercial_ok=True,
    open_vocabulary=False,
    gives_masks=False,
    pip_extra="transformers",
    notes="Real-time AND licence-clean, but fixed vocabulary. Use it to measure what "
          "open vocabulary is actually worth.",
)

# Map the model's own class names onto our detector vocabulary. Anything our
# vocabulary asks for that isn't here is unreachable for this model, by design.
COCO_O365_ALIASES = {
    "sofa": ["couch", "sofa"],
    "armchair": ["chair", "couch"],
    "coffee table": ["coffee table", "table"],
    "side table": ["side table", "table", "nightstand"],
    "television": ["tv", "tvmonitor", "monitor", "television"],
    "tv stand": ["cabinet/shelf", "cabinet"],
    "bookcase": ["cabinet/shelf", "bookcase", "book shelf"],
    "bed": ["bed"],
    "mirror": ["mirror"],
    "washing machine": ["washing machine", "washing machine/drying machine"],
    "refrigerator": ["refrigerator"],
    "microwave": ["microwave", "microwave oven"],
    "chair": ["chair"],
    "dining table": ["dining table", "table"],
    "desk": ["desk", "table"],
    "cardboard box": ["storage box", "box"],
    "door": ["door"],
}


class RtDetrDetector:
    info = INFO

    def __init__(self, threshold: float = 0.35):
        self.th = threshold
        self._proc = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self._proc = AutoImageProcessor.from_pretrained(INFO.checkpoint)
            self._model = AutoModelForObjectDetection.from_pretrained(
                INFO.checkpoint).to(pick_device()).eval()
        return self._proc, self._model

    def unreachable(self, prompts: Sequence[str]) -> list[str]:
        """Which requested prompts this model has no class for. Recorded for fairness."""
        return [p for p in prompts if p.lower() not in COCO_O365_ALIASES]

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        import cv2, torch
        from PIL import Image
        proc, model = self._load()
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=pil, return_tensors="pt").to(pick_device())
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(
            out, target_sizes=torch.tensor([pil.size[::-1]], device=pick_device()),
            threshold=self.th)[0]

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        # invert the alias map: model class name -> our prompt
        rev: dict[str, str] = {}
        for ours, theirs in COCO_O365_ALIASES.items():
            for t in theirs:
                rev.setdefault(t.lower(), ours)

        wanted = {p.lower() for p in prompts}
        dets = []
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            raw = id2label.get(int(l), "")
            ours = rev.get(raw)
            if ours is None or ours not in wanted:
                continue                       # a class we don't care about
            dets.append(Detection(label=ours, score=float(s),
                                  box=[float(v) for v in b], source=INFO.key))
        return nms(dets)
