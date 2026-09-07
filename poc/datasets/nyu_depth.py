"""
FILE PURPOSE
    Loads NYU Depth V2 — the only dataset here with REAL depth ground truth, measured
    by a Kinect rather than predicted by a model.

WHY THIS MATTERS MORE THAN HOMEOBJECTS-3K
    HomeObjects-3K has boxes but no depth, so it can only score stage 2: did the
    detector find the object? It cannot tell you whether a dimension is right, which
    is the half of the pipeline that decides the quote.

    NYU carries per-pixel depth from a depth sensor, so it can score the measurement
    chain:
      stage 3  is MoGe-2's metric depth actually as good as published, on rooms?
      stage 4  does the scale anchor IMPROVE true metric error, or degrade it?
               Right now that is assumed, never tested.
      stage 5  given true depth, does extent() recover an object's real dimensions?

    It also carries semantic and instance masks, so an object's pixels can be isolated
    without running our own detector — which separates "our measurement is wrong" from
    "our detection was wrong".

WHY NOT ULTRALYTICS' DEPTH MODEL
    Because a model is not ground truth. Comparing our depth to YOLO26-depth's depth
    measures AGREEMENT, not accuracy: both are monocular models trained on overlapping
    data with correlated biases, so they can be confidently wrong together. NYU is what
    YOLO26-depth is itself benchmarked against, and we can use it directly — no
    Ultralytics code, so no AGPL question either.

WHAT IT STILL CANNOT DO — READ THIS BEFORE TRUSTING A RESULT
    1. A Kinect cannot see the back of a wardrobe either. Single-view occlusion (gap
       B8) is a property of the viewpoint, not of the sensor, so NYU cannot validate
       the depth of a flat-fronted object.
    2. Packed volume is not measured volume. A rug measures 3 cm high lying flat and
       ships rolled. NYU has no packing information, so gap A1 NARROWS but does not
       close: hand-measured rooms remain the only validation of a quote.
    3. Kinect depth has its own errors — roughly 0.5-10 m range, holes on dark, shiny
       and transparent surfaces. Treat it as a good reference, not as truth.
    4. **NYU is US buildings.** A US interior door is 80 in = 2.032 m, against the UK
       1.981 m that door_scale() assumes. So NYU can test whether anchoring WORKS as a
       mechanism, but the 2.6% constant difference must be handled or the test will
       show a false bias. Pass door_height_m=2.032 for NYU footage.

DEPTH UNITS — verified, not assumed
    The `depth` image is uint16 in MILLIMETRES. Checked against six val images: raw
    713-7317 becomes 0.71-7.32 m, which is plausible for indoor rooms. The loader
    re-checks this on every load and raises if a file falls outside 0.3-12 m, because
    a silent unit change would corrupt every comparison downstream.

SOURCE AND LICENCE
    HF `jagennath-hari/nyuv2` — the 1,449 densely-labelled subset as parquet,
    1014 train / 290 val / 145 test, 1.28 GB.
    **This mirror states no licence.** The original NYU Depth V2 is published for
    research use; the widely-used preprocessed copy states MIT. Confirm the terms
    before any result leaves BJIT. Tracked as a licence question, like every other
    dataset and model here.

HOW TO RUN
    python -m poc.datasets.nyu_depth --download 60      # cache 60 val images
    python -m poc.datasets.nyu_depth                    # summarise the cache

USED BY  a depth-evaluation runner (not yet written)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "nyu_depth_v2"
HF_ID = "jagennath-hari/nyuv2"

MM_PER_M = 1000.0
PLAUSIBLE_M = (0.3, 12.0)     # an indoor Kinect scene. Outside this, distrust the units.

# The masks are NOT raw ids. This mirror saved them as 16-bit PNGs rescaled to fill
# the uint16 range, so a semantic value is class_id * (65535/40) and an instance value
# is instance_index * (65535/37). Passing those through unchanged looks like data and
# is not: it segments correctly (distinct value = distinct object) while every class
# id is wrong by a factor of 1638.
#
# Established empirically, not guessed: dividing by these steps leaves whole numbers to
# within 0.0005 across the cached split, and the semantic ids then land in 0-40, which
# identifies the taxonomy as NYU-40.
SEMANTIC_STEP = 65535 / 40
INSTANCE_STEP = 65535 / 37

# NYU-40, the standard reduced taxonomy. Index is the class id; 0 is unlabelled.
NYU40 = [
    "unlabelled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
    "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books",
    "refridgerator", "television", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub", "bag",
    "otherstructure", "otherfurniture", "otherprop",
]

# NYU-40 -> our detector prompts, so the same scoring code can be reused.
# 14 of our 26 prompts are reachable here, against 8 on HomeObjects-3K.
# None means "present in NYU but we do not quote it" (walls, floors) or "too ambiguous
# to map" — `cabinet` could be a kitchen unit or a sideboard, and guessing would
# manufacture false positives.
TO_OURS: dict[str, str | None] = {
    "bed": "bed", "chair": "chair", "sofa": "sofa", "door": "door",
    "table": "dining table",          # NYU-40 merges dining/coffee/side. See note below.
    "bookshelf": "bookcase", "desk": "desk", "dresser": "chest of drawers",
    "television": "television", "lamp": "floor lamp", "night stand": "side table",
    "refridgerator": "refrigerator", "box": "cardboard box", "mirror": "mirror",
    # deliberately unmapped
    "wall": None, "floor": None, "ceiling": None, "window": None, "picture": None,
    "counter": None, "blinds": None, "shelves": None, "curtain": None, "pillow": None,
    "floor mat": None, "clothes": None, "books": None, "paper": None, "towel": None,
    "shower curtain": None, "whiteboard": None, "person": None, "toilet": None,
    "sink": None, "bathtub": None, "bag": None, "cabinet": None,
    "otherstructure": None, "otherfurniture": None, "otherprop": None,
    "unlabelled": None,
}

SCORABLE_PROMPTS = sorted({v for v in TO_OURS.values() if v})

# `table` is a known compromise: NYU-40 has one `table` where our cube table has
# coffee_table (0.34 m3), side_table (0.17) and dining tables. Scoring a detected
# "coffee table" against a ground-truth "dining table" would count a correct detection
# as wrong. Treat table recall on NYU as a lower bound, not a measurement.


def download(limit: int = 60, split: str = "val", force: bool = False) -> Path:
    """Stream `limit` examples and cache them as .npz — rgb, depth in metres, masks.

    Streaming on purpose: the parquet shards are 1.28 GB and we need tens of images,
    not fifteen hundred. Caching as .npz means later runs need no network and no
    `datasets` dependency at all.
    """
    out = CACHE / split
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("*.npz"))
    if len(have) >= limit and not force:
        print(f"{len(have)} examples already cached in {out}. --force to redo.")
        return out

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("needs the `datasets` package:\n"
                         "    uv pip install --python poc/.venv/bin/python datasets")

    print(f"streaming {limit} {split} examples from {HF_ID}")
    ds = load_dataset(HF_ID, split=split, streaming=True)
    n = 0
    for ex in ds:
        if n >= limit:
            break
        rgb = np.array(ex["rgb"])
        depth_mm = np.array(ex["depth"]).astype(np.float32)
        depth_m = depth_mm / MM_PER_M
        _assert_units(depth_m, str(ex.get("id", n)))
        np.savez_compressed(
            out / f"{str(ex.get('id', n)).zfill(5)}.npz",
            rgb=rgb, depth_m=depth_m.astype(np.float32),
            semantic=np.array(ex["semantic"]).astype(np.uint16),
            instance=np.array(ex["instance"]).astype(np.uint16))
        n += 1
        print(f"  {n}/{limit}", end="\r", flush=True)
    print(f"\ncached {n} examples in {out}")
    return out


def _assert_units(depth_m: np.ndarray, ident: str) -> None:
    """Fail loudly if depth is not in the range a room should occupy.

    The units are documented nowhere on the source, so they were established
    empirically. If the mirror ever changes them, this raises instead of quietly
    scaling every measurement by 1000.
    """
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        raise ValueError(f"{ident}: depth map is entirely zero")
    lo, hi = float(valid.min()), float(valid.max())
    if not (PLAUSIBLE_M[0] <= lo and hi <= PLAUSIBLE_M[1]):
        raise ValueError(
            f"{ident}: depth reads {lo:.2f}-{hi:.2f} m after dividing by "
            f"{MM_PER_M:.0f}, outside the plausible {PLAUSIBLE_M[0]}-{PLAUSIBLE_M[1]} m. "
            "The source's units have probably changed — do NOT scale past this.")


def load(split: str = "val", limit: int | None = None) -> list[dict]:
    """Return [{id, rgb (BGR uint8), depth_m, semantic, instance}].

    RGB comes back as **BGR**, because every adapter in poc/models takes BGR — the
    same convention cv2.imread uses. Converting here rather than at each call site
    keeps one place responsible for it.
    """
    d = CACHE / split
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"nothing cached in {d}. Run:\n"
            f"    python -m poc.datasets.nyu_depth --download 60 --split {split}")
    out = []
    for f in files[:limit] if limit else files:
        z = np.load(f)
        depth = z["depth_m"]
        _assert_units(depth, f.name)
        sem = np.round(z["semantic"].astype(np.float64) / SEMANTIC_STEP).astype(np.uint8)
        inst = np.round(z["instance"].astype(np.float64) / INSTANCE_STEP).astype(np.uint8)
        out.append(dict(id=f.stem, rgb=z["rgb"][:, :, ::-1].copy(),
                        depth_m=depth,
                        semantic=sem,                 # NYU-40 class ids, 0 = unlabelled
                        instance=inst,                # per-image instance index
                        classes={int(i): NYU40[int(i)] for i in np.unique(sem)
                                 if int(i) < len(NYU40)}))
    return out


def summary(split: str = "val") -> dict:
    items = load(split)
    ranges = []
    for it in items:
        v = it["depth_m"][it["depth_m"] > 0]
        ranges.append((float(v.min()), float(v.max()), float(np.median(v))))
    lo = [r[0] for r in ranges]; hi = [r[1] for r in ranges]; md = [r[2] for r in ranges]
    return {
        "split": split, "images": len(items),
        "resolution": list(items[0]["rgb"].shape[:2][::-1]),
        "depth_m": {"nearest": round(min(lo), 2), "furthest": round(max(hi), 2),
                    "median_of_medians": round(float(np.median(md)), 2)},
        "instances_per_image_median": int(np.median(
            [len(np.unique(it["instance"])) - 1 for it in items])),
        "semantic_classes_seen": sorted({n for it in items for n in it["classes"].values()
                                         if n != "unlabelled"}),
        "our_prompts_reachable": sorted({TO_OURS[n] for it in items
                                         for n in it["classes"].values()
                                         if TO_OURS.get(n)}),
        "has_real_depth": True,
        "has_object_masks": True,
        "taxonomy": "NYU-40 (decoded from a uint16 rescale — see SEMANTIC_STEP)",
        "caveat": "US doors are 2.032 m, not the UK 1.981 m door_scale() assumes. "
                  "Single-view occlusion still hides the back of objects. "
                  "No packed-volume labels, so gap A1 narrows but stays open.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("WHY")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", type=int, metavar="N",
                    help="stream and cache N examples")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.download:
        download(a.download, a.split, a.force)
    print(json.dumps(summary(a.split), indent=2))
    print("\nReal depth here scores stages 3-5. Detection still scores against "
          "HomeObjects-3K,\nand only a tape measure scores a QUOTE (gap A1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
