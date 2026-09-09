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

HOW TO RUN
    Not an entry point - this defines the contracts and shared helpers.
    To see what implements them:  python -m poc.models.registry
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
    label_raw: str = ""                   # the detector's own text, pre-normalisation
    label_reason: str = ""                # normalise_label() verdict: "", narrowed,
                                          # ambiguous, empty, unmatched
    label_candidates: tuple[str, ...] = ()  # when ambiguous, the prompts that tied.
                                          # Consumers should search the UNION of their
                                          # size classes rather than all 42.

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
    """Best available torch device, unless overridden.

    Set POC_DEVICE to force one:  export POC_DEVICE=cpu

    Why the override exists: on Apple Silicon some PyTorch ops have no Metal kernel.
    Usually PYTORCH_ENABLE_MPS_FALLBACK=1 handles it by running those on the CPU, but
    a few still hard-fail. Forcing cpu for one model is faster than debugging Metal.
    See RUNNING.md.
    """
    import os
    import torch
    forced = os.environ.get("POC_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        if forced == "cuda" and not torch.cuda.is_available():
            print("  ! POC_DEVICE=cuda but no CUDA device; falling back to auto")
        elif forced == "mps" and not torch.backends.mps.is_available():
            print("  ! POC_DEVICE=mps but MPS unavailable; falling back to auto")
        else:
            return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_label(raw: str, prompts: Sequence[str]) -> tuple[str, str]:
    """Map a detector's raw text onto exactly one vocabulary prompt.

    WHY THIS EXISTS — measured against transformers 5.16.1 / grounding-dino-base,
    not theorised. Grounding DINO does not return a class name. It returns the
    TOKEN SPAN whose scores beat `text_threshold`, and that span changes shape
    with the threshold:

        threshold too high  ->  ''                          no token qualified
        threshold too low   ->  'sofa wardrobe door chair'   several qualified
        inside the band     ->  'sofa'                       what we actually want

    Both off-nominal forms are correct model behaviour, and both used to damage
    us in three separate places:

      1. detect_vocab.json lookup misses, so `allowed` becomes None and the
         size-class search widens from 2-3 candidates to all 42. nearest_class()
         puts that at roughly 18 points of accuracy — lost silently, with no
         error and nothing in the output to show it happened.
      2. `"door" in label` matched the span 'sofa wardrobe door chair', so a real
         sofa was discarded as if it were the scale anchor.
      3. nms() groups by exact label, so 'sofa wardrobe chair' and
         'sofa wardrobe door chair' counted as different objects and never
         suppressed each other. The same sofa was counted twice and the volume
         over-estimated — which is the precise failure this project exists to fix.

    Returns (label, reason), where reason is one of:
        ""           exact match on a vocabulary prompt — the good case
        "narrowed"   exactly one prompt found inside a longer span, safe
        "ambiguous"  several prompts fused; label is "" ON PURPOSE, see below
        "empty"      the model named nothing
        "unmatched"  text that matches no prompt at all

    In every case where we cannot identify ONE prompt, the label is "" rather
    than a guess. An unnamed box is still a real object worth measuring, and
    Method A's classifier can name it later — that is the intended division of
    labour ("the detector counts, the chat model names"). Inventing a class is
    strictly worse than admitting we do not know; the "ambiguous" branch below
    explains exactly why.
    """
    s = " ".join(str(raw).split()).lower()
    if not s:
        return "", "empty", ()
    known = [p.lower() for p in prompts]
    if s in known:
        return s, "", (s,)
    hits = [p for p in known if p in s]
    # 'chair' is a substring of 'armchair'. A nested pair is one object, not two
    # competing readings, so drop any hit subsumed by a longer hit before we
    # decide whether the span is genuinely ambiguous.
    hits = [h for h in hits if not any(h != o and h in o for o in hits)]
    if len(hits) == 1:
        return hits[0], "narrowed", (hits[0],)
    if hits:
        # Several distinct prompts fused into one span. We return no single label —
        # but we DO return the candidates, and callers must search the union of
        # their size classes. That matters, and a real run proved it:
        #
        #   span 'coffee table side table' on an actual coffee table
        #     -> ambiguous -> unnamed -> nearest_class searched all 42 by geometry
        #     -> chose sofa_3_seat, 1.416 m3, for an object of roughly 0.3 m3
        #
        # Both candidates were tables that AGREE on size, so collapsing to nothing
        # threw away information we had. Searching {coffee_table, side_table,
        # bedside_table} would have kept it. Hence label_candidates.
        #
        # We still refuse to invent ONE label, because a wrong single label is
        # worse than no label:
        #
        #   no label     -> nearest_class searches all 42 classes and picks by
        #                   MEASURED w/d/h. Geometry-driven, degrades gracefully.
        #   wrong label  -> nearest_class is FORCED into that class's 2-3
        #                   candidates whatever the tape measure says. Calling a
        #                   sofa a 'cardboard box' pins 1.5 m3 of object to a
        #                   0.06 m3 class, and the error is unrecoverable.
        #
        # An earlier version of this function returned max(hits, key=len) — the
        # longest matching prompt. On a real span of
        #   'sideboard rolled rug mattress cardboard box door'
        # that produced 'cardboard box', which won on string length alone and
        # carried no semantic claim whatsoever. Confident nonsense is the one
        # output this pipeline must never produce.
        return "", "ambiguous", tuple(sorted(hits))
    return "", "unmatched", ()


def nms(dets: list[Detection], iou_thresh: float = 0.65) -> list[Detection]:
    """Greedy non-maximum suppression, per label, plus an unnamed-duplicate pass.

    Different detectors disagree about how many boxes to emit for one object.
    Normalising that here keeps the comparison about the models rather than about
    their default post-processing.

    Two passes, for the reason spelled out at the second one: per-label first,
    then a cross-label sweep that removes unnamed boxes shadowing named ones.
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
    # sorted(), not a bare set. Iterating `{d.label for d in dets}` walks a set of
    # STRINGS, and Python randomises string hashing per process (PYTHONHASHSEED), so
    # the output order changed on every run. The volume totals were unaffected — they
    # are order-independent sums — but the `measurements` rows in every results JSON
    # came out shuffled, which makes two runs impossible to diff and silently breaks
    # any comparison that pairs rows by position. For a project whose whole method is
    # comparing combinations, reproducible ordering is not cosmetic.
    for label in sorted({d.label for d in dets}):
        group = sorted([d for d in dets if d.label == label], key=lambda d: -d.score)
        keep: list[Detection] = []
        for d in group:
            if all(iou(d.box, k.box) < iou_thresh for k in keep):
                keep.append(d)
        out.extend(keep)

    # Second pass, across labels, for unnamed boxes only.
    #
    # Grouping by label is right for named objects — a chair in front of a sofa
    # genuinely is two overlapping objects. But normalise_label() deliberately
    # emits "" when it will not guess a class, and an unnamed box sitting on top
    # of a NAMED box is not a second object: it is the same object, detected
    # twice, where one copy failed to clear text_threshold. Per-label grouping
    # cannot see that, so both survived and the object was counted twice —
    # inflating volume, the exact failure this project exists to fix.
    #
    # The named box strictly dominates: same geometry, plus a class. So drop the
    # unnamed duplicate and keep the named one.
    named = [d for d in out if d.label]
    if named:
        out = [d for d in out
               if d.label or all(iou(d.box, n.box) < iou_thresh for n in named)]
    # Final total order: strongest first, then by geometry to break ties. Deterministic
    # for a given input regardless of process, so results JSONs diff cleanly.
    out.sort(key=lambda d: (-d.score, d.box[0], d.box[1], d.box[2], d.box[3], d.label))
    return out
