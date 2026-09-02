"""
FILE PURPOSE
    The contracts every model adapter must satisfy. This file is what makes models
    swappable: as long as an adapter returns these shapes, the pipeline neither knows
    nor cares which model produced them.

WHY IT MATTERS
    We want to compare many detector x depth x classifier combinations. Without one
    uniform output shape, every combination needs its own bespoke pipeline and the
    comparison becomes impossible to trust. One contract, many implementations.

WHAT'S HERE
    Detection      one detected object: label, confidence, box, optional mask
    DepthResult    a metric point map plus validity mask and intrinsics
    Detector       protocol: .detect(image, prompts) -> list[Detection]
    DepthEstimator protocol: .infer(image) -> DepthResult
    Classifier     protocol: .classify(image, allowed) -> list[ClassifiedItem]
    Segmenter      protocol: .refine(image, boxes) -> masks   (box -> mask upgrade)

USED BY  every file in poc/models/, and poc/runner/pipeline.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable
import numpy as np


# --------------------------------------------------------------------------- data


@dataclass
class Detection:
    """One object found in one frame.

    box    [x0, y0, x1, y1] in pixels, top-left origin.
    mask   Optional boolean array the size of the frame. Present only when the
           adapter is a segmentation model or a Segmenter has refined the box.
           THIS MATTERS: a box around a sofa also contains wall and floor pixels,
           so measuring "everything in the box" over-measures the object. A mask
           removes that background and is the single cheapest accuracy win
           available to Method B.
    """
    label: str
    score: float
    box: list[float]
    mask: np.ndarray | None = None
    source: str = ""                      # which adapter produced this

    @property
    def area_px(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class DepthResult:
    """Metric geometry for one frame.

    points  (H, W, 3) X/Y/Z in METRES, camera coordinates
    depth   (H, W)    distance from the camera plane, in metres
    mask    (H, W)    True where the model is confident
    vert_axis  which axis of `points` is vertical (floor-to-ceiling). Adapters set
               this because conventions differ between models, and the scale anchor
               depends on getting it right.
    """
    points: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray | None = None
    vert_axis: int = 1
    source: str = ""


@dataclass
class ClassifiedItem:
    """One item named by a Method A classifier."""
    size_class: str
    count: int
    confidence: float
    reasoning: str = ""


@dataclass
class ModelInfo:
    """Everything the registry needs to know about an adapter without importing it.

    commercial_ok is the field that matters most. It is not advisory — the runner
    refuses to run a non-commercial model unless explicitly overridden, because
    building a prototype on a model that can never ship wastes the work.
    """
    key: str
    kind: str                             # "detector" | "depth" | "classifier" | "segmenter"
    display: str
    checkpoint: str
    licence: str
    commercial_ok: bool
    open_vocabulary: bool = False
    gives_masks: bool = False
    pip_extra: str | None = None          # what to install if missing
    notes: str = ""
    warn: str = ""                        # printed loudly before use


# ----------------------------------------------------------------------- protocols


@runtime_checkable
class Detector(Protocol):
    info: ModelInfo
    def detect(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Detection]: ...


@runtime_checkable
class DepthEstimator(Protocol):
    info: ModelInfo
    def infer(self, image_bgr: np.ndarray) -> DepthResult: ...


@runtime_checkable
class Classifier(Protocol):
    info: ModelInfo
    def classify(self, image_bgr: np.ndarray, allowed: Sequence[str]) -> tuple[list[ClassifiedItem], dict]: ...


@runtime_checkable
class Segmenter(Protocol):
    info: ModelInfo
    def refine(self, image_bgr: np.ndarray, boxes: Sequence[list[float]]) -> list[np.ndarray]: ...


# ------------------------------------------------------------------------- helpers


def pick_device():
    """Best available torch device. Imported lazily so this module stays light."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def nms(dets: list[Detection], iou_thresh: float = 0.65) -> list[Detection]:
    """Greedy non-maximum suppression, per label.

    Different detectors disagree about how many boxes to emit for one object.
    Normalising that here keeps the comparison about the models rather than about
    their default post-processing.
    """
    def iou(a: list[float], b: list[float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    out: list[Detection] = []
    for label in {d.label for d in dets}:
        group = sorted([d for d in dets if d.label == label], key=lambda d: -d.score)
        keep: list[Detection] = []
        for d in group:
            if all(iou(d.box, k.box) < iou_thresh for k in keep):
                keep.append(d)
        out.extend(keep)
    return out
