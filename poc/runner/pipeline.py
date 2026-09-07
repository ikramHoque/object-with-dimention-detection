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
    discover_rooms   input folder -> the rooms in it, named           (stage 1)
    room_name_for    any --input -> which room it belongs to          (stage 1)
    load_keyframes   video/image/folder -> sharp frames               (stage 1)
    measure_room     ONE room, all of stages 1-5, shared by every caller
    door_scale       correction factor from a known-height door       (stage 4)
    extent           3D size of one object, mask-aware                (stage 5)
    nearest_class    measured size -> closest cube-table entry        (stage 5)
    dedup_counts     per-frame counts -> one room inventory           (stage 7)
    vol_from_inventory  inventory -> cubic metres                     (stage 8)

USED BY  poc/runner/run_combination.py and every poc/pipelines/*/notebook.ipynb

HOW TO RUN
    Not an entry point - this is a library of stage functions.
    It is imported by run_combination.py, run_dataset_eval.py and the notebook.
    To run the pipeline:  python -m poc.runner.run_combination --help
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


def discover_rooms(in_dir: Path) -> list[dict]:
    """What rooms are in an input folder, and what each one is called.

    THE RULE: a SUB-FOLDER IS A ROOM, and its folder name IS the room name.

        data/input/
            bedroom/        -> room "bedroom"  (every image inside is a view of it)
                north.jpg
                south.jpg
            lounge/         -> room "lounge"
            kitchen/        -> room "kitchen"
            stray.jpg       -> room "stray"    (a loose file is a one-photo room)

    So four folders means a four-room house, and the room name travels with every
    number the pipeline produces — into the JSON, the CSV, the annotated frame and
    the filename.

    WHY FOLDERS AND NOT LOOSE FILES. Stages 7-8 count each item once PER ROOM: a
    sofa photographed from three angles is one sofa. That only works if the code
    knows which photographs belong together, and a folder is how you tell it. Eight
    loose photos of eight different rooms would otherwise merge into one inventory
    for a house that does not exist.

    A loose file is still accepted, as one room named after the file, because
    checking a single photograph is a thing you do constantly while tuning.

    Returns dicts with name/path/kind/n_images, ordered folders-first then by name,
    so the listing you see matches the order things run in.
    """
    if not in_dir.is_dir():
        return []
    rooms = []
    for q in sorted(in_dir.iterdir(), key=lambda x: x.name.lower()):
        if q.name.startswith(".") or q.name == "__pycache__":
            continue
        if q.is_dir():
            n = sum(1 for f in q.iterdir() if f.suffix.lower() in IMG_EXT)
            if n:
                rooms.append(dict(name=q.name, path=q, kind="folder", n_images=n))
        elif q.suffix.lower() in IMG_EXT:
            rooms.append(dict(name=q.stem, path=q, kind="photo", n_images=1))
        elif q.suffix.lower() in VID_EXT:
            rooms.append(dict(name=q.stem, path=q, kind="video", n_images=0))
    rooms.sort(key=lambda r: (r["kind"] != "folder", r["name"].lower()))
    return rooms


def room_name_for(src: Path, in_dir: Path) -> str:
    """The room name for anything you can point --input at.

    A single image inside a room folder still belongs to that ROOM, so its parent
    folder names it — otherwise `--input lounge/north.jpg` would file its results
    under "north", and the room it actually measured would be lost from the report.

        input/lounge                -> "lounge"      the whole room
        input/lounge/north.jpg      -> "lounge"      one view of that room
        input/stray.jpg             -> "stray"       a loose photo
        /somewhere/else/hall.jpg    -> "hall"        outside data/input entirely
    """
    src, in_dir = Path(src), Path(in_dir)
    if src.is_dir():
        return src.name
    # A file's room is the sub-folder holding it — but only when that folder is a
    # room, i.e. strictly BELOW data/input. A file sitting directly in data/input,
    # or anywhere outside it, is named by the file itself.
    parent = src.parent
    try:
        below_in_dir = in_dir.resolve() in parent.resolve().parents
    except OSError:                 # a path we cannot resolve — fall back to the file
        below_in_dir = False
    return parent.name if (below_in_dir and parent.name) else src.stem


def load_keyframes(path: Path, *, stride_s: float = 1.0, max_frames: int = 16,
                   blur_min: float = 60.0, bright_range=(35, 225),
                   long_edge: int = 1024):
    """Read an image, a video, or a FOLDER of stills; return (all_sampled, kept).

    A folder is treated as several views of ONE room, which is the real product
    shape: a customer photographs a bedroom from six angles rather than filming
    it. Frames are ordered by filename and given synthetic timestamps, so the
    dedup stage sees them exactly as it would see video keyframes.

    IMPORTANT — one folder is one ROOM, not one dataset. Stages 7-8 deduplicate
    across frames, so pointing this at 50 photos of 50 different rooms would
    merge them into a single nonsense inventory. One room per folder, one run
    per room.
    """
    import cv2
    raw = []
    if path.is_dir():
        files = sorted(q for q in path.iterdir() if q.suffix.lower() in IMG_EXT)
        if not files:
            raise IOError(f"no images ({', '.join(sorted(IMG_EXT))}) in {path}")
        for i, q in enumerate(files):
            img = cv2.imread(str(q))
            if img is None:
                print(f"  ! skipping unreadable {q.name}")
                continue
            raw.append((float(i), img))
        if not raw:
            raise IOError(f"no readable images in {path}")
    elif path.suffix.lower() in IMG_EXT:
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
        raise ValueError(
            f"unsupported input: {path.name}. Pass an image "
            f"({', '.join(sorted(IMG_EXT))}), a video "
            f"({', '.join(sorted(VID_EXT))}), or a folder of stills of one room.")

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


def allowed_classes(det, vocab: dict) -> list[str] | None:
    """Which cube-table classes may this detection be matched against?

    Restricting the search is worth roughly 18 points of accuracy, so it is worth
    being careful about. Three cases, in order of how much the label tells us:

      clean / narrowed label  -> that prompt's own 2-3 classes
      ambiguous label         -> the UNION of the tied candidates' classes
      nothing usable          -> None, meaning search the whole table by geometry

    The middle case exists because collapsing ambiguity to nothing threw away real
    information. On a live run the span 'coffee table side table' became unnamed and
    geometry then matched an actual coffee table to sofa_3_seat: 1.416 m3 against a
    true ~0.3 m3. Both candidates were tables that agreed on size, so the union
    {bedside_table, coffee_table, side_table} was available the whole time.

    Returns None rather than [] for "no restriction", because nearest_class treats a
    falsy value as "search everything" and an empty list would silently match nothing.
    """
    direct = vocab.get(det.label)
    if direct:
        return list(direct)
    cands = getattr(det, "label_candidates", ()) or ()
    union = sorted({c for prompt in cands for c in vocab.get(prompt, [])})
    return union or None


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


# ------------------------------------------------ stages 1-5, for ONE room


def measure_room(src, det, dep, seg=None, *, prompts, classes, vocab,
                 max_frames: int = 16, use_anchor: bool = True, log=None):
    """Ingest, detect, depth, anchor and measure ONE room. Stages 1-5.

    WHY THIS IS A FUNCTION AND NOT COPIED INTO EACH CALLER
        run_combination.py and every pipeline notebook both need exactly this loop.
        It used to exist twice, and the two copies had already drifted: the notebook
        wrote det_label='(unnamed)' where the CLI wrote '', which silently broke the
        colour coding in the annotated picture — report._colour() treats a non-empty
        label as named, so unnamed boxes were drawn GREEN ("measured and named")
        instead of RED ("we do not know what this is"). A reader would have trusted
        a box the pipeline could not identify.

        One copy means the notebook and the CLI cannot disagree, which is the whole
        claim the notebooks make about themselves.

    STAGE 6 IS NOT HERE. The classifier is a second, independent opinion and the
    caller decides whether to pay for it. Stages 7-9 are two lines each and differ
    per caller. This is the expensive, fiddly middle.

    det_label is '' for a box we could not name — never a placeholder string.
    Display code turns it into '(unnamed)'; comparisons need the empty string.

    log: called with one line at a time so a long run shows progress. None is silent.
    """
    import time
    say = log or (lambda _m: None)
    timings: dict[str, float] = {}

    # ---- stage 1 ingest
    t0 = time.time()
    sampled, frames = load_keyframes(src, max_frames=max_frames)
    timings["ingest_s"] = round(time.time() - t0, 2)
    say(f"stage 1  {len(frames)}/{len(sampled)} frames kept  ({timings['ingest_s']}s)")
    if not frames:
        return dict(sampled=sampled, frames=[], rows=[], scale=1.0,
                    scale_src="no usable frames", factors=[], whys=[], n_det=0,
                    unreachable=[], masked=0, prior=0, all_dets=[], reasons={},
                    lab_missing=0, lab_fixed=0, timings=timings)

    # ---- stage 2 detect (+ optional mask refinement)
    t0 = time.time()
    for f in frames:
        f["dets"] = det.detect(f["img"], prompts)
        if seg is not None:
            for d, m in zip(f["dets"], seg.refine(f["img"], [x.box for x in f["dets"]])):
                d.mask = m
    timings["detect_s"] = round(time.time() - t0, 2)
    n_det = sum(len(f["dets"]) for f in frames)
    unreachable = det.unreachable(prompts) if hasattr(det, "unreachable") else []
    say(f"stage 2  {n_det} detections  ({timings['detect_s']}s)")

    # ---- stage 3 depth
    t0 = time.time()
    for f in frames:
        f["depth"] = dep.infer(f["img"])
    timings["depth_s"] = round(time.time() - t0, 2)
    say(f"stage 3  depth done  ({timings['depth_s']}s)")

    # ---- stage 4 scale anchor
    factors, whys = [], []
    for f in frames:
        fac, why = door_scale(f["dets"], f["depth"])
        whys.append(why)
        if fac:
            factors.append(fac)
    if not use_anchor:
        scale, scale_src = 1.0, "anchor disabled"
    elif not factors:
        scale, scale_src = 1.0, "no door found"
    else:
        # Median, not mean: one badly-cropped door in eight frames should not drag
        # the whole room's scale with it.
        scale = float(np.median(factors))
        scale_src = f"door anchor, median of {len(factors)} frames"
    say(f"stage 4  SCALE={scale:.4f}  [{scale_src}]")

    # ---- stage 5 measure (Method B)
    t0 = time.time()
    rows = []
    for f in frames:
        # (Detection, row) pairs per frame, so report.annotate() can put the measured
        # dimensions on the right box. Built here rather than matched by index later —
        # row order and detection order diverge the moment a box is skipped.
        f["pairs"] = []
        for d in f["dets"]:
            # Exact match, not `"door" in d.label`. Labels are normalised to a single
            # vocabulary prompt, and the old substring test matched the raw span
            # 'sofa wardrobe door chair', throwing away a real sofa as the anchor.
            if d.label == "door":
                continue
            e = extent(f["depth"], d, scale=scale)
            if not e:
                continue
            # resolve_dims handles the single-view depth limit: a camera never sees the
            # back of a wardrobe, so for flat-fronted objects the depth extent is
            # unobservable and a class prior is substituted (gap B8). allowed_classes
            # narrows the size-class search as tightly as the label honestly permits.
            dims, cls, dist, depth_src = resolve_dims(e, classes, allowed_classes(d, vocab))
            row = dict(t=f["t"], det_label=d.label, score=round(d.score, 3),
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in dims.items()},
                       mapped_class=cls, map_dist=round(dist, 3),
                       depth_source=depth_src)
            rows.append(row)
            f["pairs"].append((d, row))
    timings["measure_s"] = round(time.time() - t0, 2)

    # ---- label quality: a tuning signal, not an error count
    all_dets = [d for f in frames for d in f["dets"]]
    reasons: dict[str, int] = {}
    for d in all_dets:
        r = getattr(d, "label_reason", "") or "clean"
        reasons[r] = reasons.get(r, 0) + 1
    masked = sum(1 for r in rows if r.get("used_mask"))
    prior = sum(1 for r in rows if r.get("depth_source") == "class_prior")
    say(f"stage 5  {len(rows)} objects measured, {masked} using masks")

    return dict(sampled=sampled, frames=frames, rows=rows, scale=scale,
                scale_src=scale_src, factors=factors, whys=whys, n_det=n_det,
                unreachable=unreachable, masked=masked, prior=prior,
                all_dets=all_dets, reasons=reasons,
                lab_missing=sum(1 for d in all_dets if not d.label),
                lab_fixed=reasons.get("narrowed", 0), timings=timings)


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
