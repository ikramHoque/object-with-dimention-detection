"""
FILE PURPOSE
    Detector adapter · YOLOE ("Real-Time Seeing Anything"), via Ultralytics.

WHAT IT IS
    The newer attempt to keep promptable open-vocabulary behaviour inside the
    real-time YOLO regime. Supports three prompting modes — text, visual exemplar,
    and prompt-free — and the segmentation variants return MASKS, which is the
    property Method B actually wants.

WHY INCLUDE IT
    If it holds YOLO speed while matching open-vocabulary accuracy AND giving masks,
    it is the best of all worlds on paper. Worth one comparison run.

##############################################################################
#  SAME LICENCE WARNING AS YOLO-WORLD — AGPL-3.0 via Ultralytics.
#  Registered commercial_ok=False. Comparison use only; not shippable without a
#  paid Enterprise Licence. See detect_yolo_world.py for the full explanation.
##############################################################################

HOW TO RUN
    Not an entry point. This adapter is selected by key:
        --detector yoloe
    passed to run_research_ceiling (AGPL - quarantined).
    Confirm it is installed:  python -m poc.models.registry
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, nms

INFO = ModelInfo(
    key="yoloe",
    kind="detector",
    display="YOLOE-v8L-seg (via Ultralytics)",
    checkpoint="yoloe-v8l-seg.pt",
    licence="AGPL-3.0 or paid Ultralytics Enterprise",
    commercial_ok=False,
    open_vocabulary=True,
    gives_masks=True,
    pip_extra="ultralytics",
    notes="Real-time open-vocabulary WITH masks. Best-of-both on paper. "
          "Checkpoint names move between Ultralytics releases — check their docs if "
          "loading fails.",
    warn="AGPL-3.0, same Ultralytics Enterprise requirement as YOLO-World.",
)


class YoloeDetector:
    info = INFO

    def __init__(self, conf: float = 0.10):
        self.conf = conf
        self._model = None
        self._classes: list[str] = []

    def _load(self, prompts: Sequence[str]):
        if self._model is None:
            from ultralytics import YOLOE
            self._model = YOLOE(INFO.checkpoint)
        self._classes = [p.lower() for p in prompts]
        # Text-prompt mode. Some releases want get_text_pe(), older ones just set_classes().
        try:
            self._model.set_classes(self._classes, self._model.get_text_pe(self._classes))
        except (AttributeError, TypeError):
            self._model.set_classes(self._classes)
        return self._model

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        model = self._load(prompts)
        res = model.predict(image_bgr, conf=self.conf, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        masks = None
        if getattr(res, "masks", None) is not None and res.masks.data is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)
        for i, b in enumerate(res.boxes):
            idx = int(b.cls.item())
            label = self._classes[idx] if idx < len(self._classes) else str(idx)
            dets.append(Detection(label=label, score=float(b.conf.item()),
                                  box=[float(v) for v in b.xyxy[0].tolist()],
                                  mask=(masks[i] if masks is not None and i < len(masks) else None),
                                  source=INFO.key))
        return nms(dets)
