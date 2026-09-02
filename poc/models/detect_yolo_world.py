"""
FILE PURPOSE
    Detector adapter · YOLO-World (Tencent AI Lab), run via the Ultralytics package.

WHY YOU'D WANT IT
    Open-vocabulary detection at real YOLO speed: 35.4 AP zero-shot on LVIS at 52 FPS
    on a V100 — roughly 20x faster than Grounding DINO at about a fifth of the size,
    with competitive accuracy. If this pipeline ever runs at scale, that speed is the
    difference between a GPU bill and a rounding error.

##############################################################################
#  LICENCE WARNING — READ BEFORE USING
#
#  Ultralytics ships under AGPL-3.0. Per Ultralytics' own published position, ANY
#  use of their models — explicitly including R&D inside a company, commercial or
#  not — requires a paid Enterprise Licence unless you open-source your ENTIRE
#  project under AGPL-3.0.
#
#  AGPL is stricter than GPL: it reaches SaaS deployment too, so "we only expose
#  it as an API" is not an escape.
#
#  For NX this means: usable to generate a comparison number in the POC only if
#  BJIT/NX accept that position, and NOT shippable in a closed-source product
#  without buying the Enterprise Licence.
#
#  This is why the adapter is registered commercial_ok=False. The runner will
#  refuse it unless you pass --allow-noncommercial.
##############################################################################

ALTERNATIVE
    If the speed is what you want but AGPL is not acceptable, RT-DETR is Apache 2.0
    and real-time — see detect_rtdetr.py. It is closed-vocabulary, which is the
    trade.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import Detection, ModelInfo, nms

INFO = ModelInfo(
    key="yolo_world",
    kind="detector",
    display="YOLO-World v2-L (via Ultralytics)",
    checkpoint="yolov8l-worldv2.pt",
    licence="AGPL-3.0 or paid Ultralytics Enterprise",
    commercial_ok=False,
    open_vocabulary=True,
    gives_masks=False,
    pip_extra="ultralytics",
    notes="~20x faster than Grounding DINO, ~5x smaller, competitive accuracy. "
          "Blocked for shipping by AGPL unless an Enterprise Licence is bought.",
    warn="AGPL-3.0. Ultralytics require an Enterprise Licence even for internal R&D "
         "unless the whole project is open-sourced. Comparison use only.",
)


class YoloWorldDetector:
    info = INFO

    def __init__(self, conf: float = 0.10):
        self.conf = conf
        self._model = None

    def _load(self, prompts: Sequence[str]):
        if self._model is None:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(INFO.checkpoint)
        # YOLO-World is told its vocabulary up front, then detects only those classes.
        self._model.set_classes([p.lower() for p in prompts])
        self._classes = [p.lower() for p in prompts]
        return self._model

    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]:
        model = self._load(prompts)
        res = model.predict(image_bgr, conf=self.conf, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        for b in res.boxes:
            idx = int(b.cls.item())
            label = self._classes[idx] if idx < len(self._classes) else str(idx)
            dets.append(Detection(label=label, score=float(b.conf.item()),
                                  box=[float(v) for v in b.xyxy[0].tolist()],
                                  source=INFO.key))
        return nms(dets)
