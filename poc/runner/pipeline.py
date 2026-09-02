"""
FILE PURPOSE
    The pipeline stages, with every model choice removed. This file knows the ALGORITHM
    — sample frames, anchor the scale, measure extents, dedup, aggregate — and nothing
    about which detector or depth model produced its inputs.

WHY IT MATTERS
    This separation is what makes the comparison honest. Every combination runs through
    exactly the same arithmetic, so a difference in the result is a difference between
    models, not between two slightly different pipelines someone copy-pasted.

WHAT'S HERE
    load_keyframes   video/image -> sharp frames                      (stage 1)
    door_scale       correction factor from a known-height door       (stage 4)
    extent           3D size of one object, mask-aware                (stage 5)
    nearest_class    measured size -> closest cube-table entry        (stage 5)
    dedup_counts     per-frame counts -> one room inventory           (stage 7)
    vol_from_inventory  inventory -> cubic metres                     (stage 8)

USED BY  poc/runner/run_combination.py, poc/pipeline.ipynb
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Sequence
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


# ------------------------------------------------------------------ stage 1


def resize_long(img, long_edge: int):
    import cv2
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA) if s < 1 else img


def sharpness(gray) -> float:
    """Variance of the Laplacian. Low = blurry."""
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_keyframes(path: Path, *, stride_s: float = 1.0, max_frames: int = 16,
                   blur_min: float = 60.0, bright_range=(35, 225),
                   long_edge: int = 1024):
    """Read an image or video and return (all_sampled, kept)."""
    import cv2
    raw = []
    if path.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"could not read {path}")
        raw = [(0.0, img)]
    elif path.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * stride_s)))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                raw.append((i / fps, fr))
            i += 1
        cap.release()
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    out = []
    for t, fr in raw:
        fr = resize_long(fr, long_edge)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s, b = sharpness(g), float(g.mean())
        out.append(dict(t=t, img=fr, sharp=s, bright=b,
                        ok=(s >= blur_min and bright_range[0] <= b <= bright_range[1])))
    return out, [f for f in out if f["ok"]][:max_frames]


# ------------------------------------------------------------------ stage 4


def door_scale(dets, depth, door_height_m: float = 1.981):
    """Correction factor from the most confident door in one frame, or (None, why)."""
    doors = [d for d in dets if "door" in d.label]
    if not doors:
        return None, "no door detected"
    d = max(doors, key=lambda x: x.score)
    x0, y0, x1, y1 = [int(v) for v in d.box]
    sub = depth.points[y0:y1, x0:x1]
    m = depth.mask[y0:y1, x0:x1].astype(bool)
    if d.mask is not None:
        m = m & d.mask[y0:y1, x0:x1].astype(bool)     # prefer the mask when we have one
    if m.sum() < 200:
        return None, "door region has too few confident depth pixels"
    v = sub[..., depth.vert_axis][m]
    # Use a WIDE percentile pair here, unlike extent(). A door box is tight and
    # contains few outliers, but we need its FULL height: clipping at 2/98 shaves
    # ~4% off a linear ramp, which inflates the scale factor by ~4% and the volume
    # by ~13% -- a systematic bias in the exact term this function exists to remove.
    measured = float(np.percentile(v, 99.5) - np.percentile(v, 0.5))
    if measured < 0.3:
        return None, f"implausible door height {measured:.2f} m"
    return door_height_m / measured, f"door measured {measured:.3f} m (score {d.score:.2f})"


# ------------------------------------------------------------------ stage 5


def extent(depth, det, scale: float = 1.0, lo: float = 2, hi: float = 98):
    """Real-world w/d/h of one detection.

    Uses det.mask when present. That matters: a box around a sofa also contains wall
    and floor, so box-only measurement systematically over-reads. `used_mask` is
    recorded in the output so the comparison can show what masks were worth.
    """
    x0, y0, x1, y1 = [int(v) for v in det.box]
    sub = depth.points[y0:y1, x0:x1]
    m = depth.mask[y0:y1, x0:x1].astype(bool)
    used_mask = False
    if det.mask is not None:
        dm = det.mask[y0:y1, x0:x1].astype(bool)
        if dm.sum() >= 50:
            m = m & dm
            used_mask = True
    if m.sum() < 100:
        return None
    p = sub[m] * scale
    va = depth.vert_axis
    h = float(np.percentile(p[:, va], hi) - np.percentile(p[:, va], lo))

    # Rotate the footprint onto the object's own axes before measuring, otherwise an
    # object angled to the camera measures far too wide.
    horiz = np.delete(p, va, axis=1)
    horiz = horiz - horiz.mean(0)
    sv = None
    try:
        _, sv, vt = np.linalg.svd(horiz, full_matrices=False)
        proj = horiz @ vt.T
    except np.linalg.LinAlgError:
        proj = horiz
    w = float(np.percentile(proj[:, 0], hi) - np.percentile(proj[:, 0], lo))
    d = float(np.percentile(proj[:, 1], hi) - np.percentile(proj[:, 1], lo))
    w, d = max(w, d), min(w, d)

    # --- DEGENERACY CHECK: depth only sees VISIBLE SURFACES -------------------
    # A camera never sees the back of a wardrobe. Its point cloud is a thin sheet,
    # so the second horizontal principal axis carries almost no variance and `d`
    # collapses toward zero -- which would make the computed volume collapse too.
    # This is not noise; it is a hard limit of single-view geometry, and it makes
    # Method B UNDER-measure depth on most furniture. We flag it rather than
    # silently returning a near-zero volume. See resolve_dims() for the fallback.
    ratio = float(sv[1] / sv[0]) if sv is not None and sv[0] > 0 else 1.0
    degenerate = (ratio < 0.12) or (d < 0.05)

    return dict(w=abs(w), d=abs(d), h=abs(h), bbox_m3=abs(w * d * h),
                n_pts=int(m.sum()), used_mask=used_mask,
                depth_observed=not degenerate, pca_ratio=round(ratio, 4))


def nearest_class(dims: dict, classes: dict, allowed: Sequence[str] | None = None):
    """Closest cube-table entry to a measurement.

    `allowed` should be the classes the detector's label permits. Restricting is worth
    roughly 18 points of accuracy at realistic scale error — never search everything.
    """
    tgt = sorted([dims["w"], dims["d"], dims["h"]], reverse=True)
    best, bd = None, 1e9
    for name in (allowed or classes.keys()):
        c = classes.get(name)
        if not c:
            continue
        cand = sorted(c["typical_dims_m"], reverse=True)
        nrm = float(np.linalg.norm(cand)) or 1e-6
        dist = float(np.linalg.norm(np.array(tgt) - np.array(cand)) / nrm)
        if dist < bd:
            best, bd = name, dist
    return best, bd




def resolve_dims(e: dict, classes: dict, allowed: Sequence[str] | None = None):
    """Turn a raw extent() result into usable dimensions plus a class.

    Handles the single-view depth limitation. When extent() reports
    depth_observed=False, the object's depth was never visible to the camera, so
    matching on all three dimensions would pick whichever class happens to be
    thinnest. Instead we match on the two dimensions we DID observe (the larger
    horizontal extent and the height), then adopt that class's typical depth.

    Returns (dims, class_name, distance, depth_source) where depth_source is
    "observed" or "class_prior". Count how often it is "class_prior": if that is
    most objects, Method B is leaning on the cube table for depth and is much
    closer to Method A than it appears.
    """
    observed = e.get("depth_observed", True)
    if observed:
        cls, dist = nearest_class(e, classes, allowed)
        return dict(e), cls, dist, "observed"

    # Match on width and height only.
    tgt = sorted([e["w"], e["h"]], reverse=True)
    best, bd = None, 1e9
    for name in (allowed or classes.keys()):
        c = classes.get(name)
        if not c:
            continue
        wd, dd, hd = c["typical_dims_m"]
        cand = sorted([max(wd, dd), hd], reverse=True)
        nrm = float(np.linalg.norm(cand)) or 1e-6
        dist = float(np.linalg.norm(np.array(tgt) - np.array(cand)) / nrm)
        if dist < bd:
            best, bd = name, dist

    out = dict(e)
    if best:
        wd, dd, hd = classes[best]["typical_dims_m"]
        out["d"] = float(min(wd, dd))              # adopt the class's typical depth
        out["bbox_m3"] = abs(out["w"] * out["d"] * out["h"])
    return out, best, bd, "class_prior"


# ------------------------------------------------------------------ stage 7


def dedup_counts(per_frame: list[dict], rule: str = "max") -> dict[str, int]:
    """Collapse per-frame counts into one inventory.

    Zeros are included on purpose: a class seen in 1 of 8 frames gets a median of 0
    and is dropped, which is the false-positive filter.
    """
    if not per_frame:
        return {}
    per = []
    for inv in per_frame:
        c = defaultdict(int)
        for it in inv["items"]:
            c[it["size_class"]] += int(it["count"])
        per.append(c)
    out = {}
    for k in sorted({k for c in per for k in c}):
        series = [c.get(k, 0) for c in per]
        v = max(series) if rule == "max" else int(np.median(series))
        if v > 0:
            out[k] = v
    return out


def dedup_measured(rows: list[dict], rule: str = "max") -> dict[str, int]:
    """Method B equivalent: max simultaneous detections of a class in any one frame."""
    if not rows:
        return {}
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("mapped_class"):
            per[r["t"]][r["mapped_class"]] += 1
    out = {}
    for cls in {c for d in per.values() for c in d}:
        series = [d.get(cls, 0) for d in per.values()]
        v = max(series) if rule == "max" else int(np.median(series))
        if v > 0:
            out[cls] = v
    return out


# ------------------------------------------------------------------ stage 8


def vol_from_inventory(inv: dict[str, int], classes: dict) -> float:
    return float(sum(inv[c] * classes[c]["cube_m3"] for c in inv if c in classes))


# ------------------------------------------------------------------ stage 9


def score_inventory(pred: dict[str, int], truth: dict[str, int], classes: dict) -> dict:
    """Precision/recall on the item list, plus volume bias and absolute error."""
    keys = set(pred) | set(truth)
    tp = sum(min(pred.get(c, 0), truth.get(c, 0)) for c in keys)
    pc, tc = sum(pred.values()), sum(truth.values())
    pv = vol_from_inventory(pred, classes)
    tv = vol_from_inventory(truth, classes)
    return dict(
        precision=round(tp / pc, 3) if pc else 0.0,
        recall=round(tp / tc, 3) if tc else 0.0,
        pred_items=pc, true_items=tc,
        pred_m3=round(pv, 3), true_m3=round(tv, 3),
        vol_bias_pct=round((pv - tv) / tv * 100, 1) if tv else None,
        vol_abs_err_pct=round(abs(pv - tv) / tv * 100, 1) if tv else None,
        missed=sorted(c for c in truth if c not in pred),
        hallucinated=sorted(c for c in pred if c not in truth),
    )
