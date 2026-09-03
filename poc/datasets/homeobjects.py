"""
FILE PURPOSE
    Loader for the Ultralytics HomeObjects-3K dataset, plus the mapping from its 12 classes
    onto our detector vocabulary.

WHY THIS DATASET IS USEFUL
    It has ground-truth BOXES. That means we can measure detection accuracy without
    hand-labelling anything — which is the one part of gap A1 we can close for free.
    Crucially it includes two classes COCO does not have:
      * `wardrobe`  - core removal item, and one of the hardest to detect
      * `door`      - OUR SCALE ANCHOR. Ground-truth door boxes let us test stage 4
                      independently of whether the detector found the door.

WHAT IT CANNOT DO — read this before drawing conclusions
    No depth. No dimensions. No volume. No camera intrinsics.
    So it validates the DETECTION half of the pipeline and nothing else. A volume number
    computed from these images has nothing to be checked against. Gap A1 stays open for
    measurement; only detection is covered.
    It is also a mix of stock/web photos, not one continuous room pan, so multi-view depth
    and cross-frame dedup cannot be tested here either.

##############################################################################
#  LICENCE · AGPL-3.0 (Ultralytics)
#
#  Practical reading:
#    EVALUATING on these images (what this file does) is ordinary research use and
#    low risk - a dataset is not a program, and we neither redistribute the images
#    nor link to Ultralytics code to read them.
#
#    TRAINING a model we intend to ship on these images is a different matter.
#    Ultralytics would treat the resulting weights as derivative. Do not do it.
#    See gap C11.
#
#  We download the zip directly and read it with opencv, deliberately avoiding the
#  `ultralytics` package so no AGPL code enters the pipeline.
##############################################################################

USED BY  poc/runner/run_dataset_eval.py
CLI      python -m poc.datasets.homeobjects --download

HOW TO RUN
    python -m poc.datasets.homeobjects --download
        Fetch and extract the dataset (~390 MB), then print a summary.
    python -m poc.datasets.homeobjects
        Summary only, if already downloaded.
    python -m poc.datasets.homeobjects --split train
        Summarise the train split instead of val.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
DATA = POC / "data" / "homeobjects3k"
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"

# The dataset's own class order, as published (index = class id in the label files).
CLASSES = ["bed", "sofa", "chair", "table", "lamp", "tv",
           "laptop", "wardrobe", "window", "door", "potted plant", "photo frame"]

# Map dataset class -> the phrase we hand our detectors, and the cube-table size classes
# it could correspond to. `None` for size classes means "not cargo" (windows, doors).
#
# Note the asymmetry: the dataset is COARSER than our vocabulary. It has one `table` where
# we distinguish coffee/side/dining/desk, and one `chair` where we split dining/office/
# armchair. So a fair comparison has to score at the dataset's granularity, not ours.
TO_OURS: dict[str, dict] = {
    "bed":          {"prompt": "bed",              "classes": ["bed_single_frame", "bed_double_frame", "bed_king_frame"]},
    "sofa":         {"prompt": "sofa",             "classes": ["sofa_2_seat", "sofa_3_seat", "sofa_sectional"]},
    "chair":        {"prompt": "chair",            "classes": ["dining_chair", "office_chair", "armchair"]},
    "table":        {"prompt": "dining table",     "classes": ["dining_table_4seat", "dining_table_6seat",
                                                                "coffee_table", "side_table", "desk"]},
    "lamp":         {"prompt": "floor lamp",       "classes": ["floor_lamp"]},
    "tv":           {"prompt": "television",       "classes": ["tv_up_to_43in", "tv_over_43in"]},
    "laptop":       {"prompt": None,               "classes": []},          # too small to matter for volume
    "wardrobe":     {"prompt": "wardrobe",         "classes": ["wardrobe_single", "wardrobe_double"]},
    "window":       {"prompt": None,               "classes": []},          # structural, not cargo
    "door":         {"prompt": "door",             "classes": None},        # SCALE ANCHOR, not cargo
    "potted plant": {"prompt": None,               "classes": []},          # not in our cube table yet
    "photo frame":  {"prompt": None,               "classes": []},          # not in our cube table yet
}

# Which of our 26 prompts this dataset can possibly confirm. Anything else we detect
# cannot be scored here - absence of a label does not mean absence of the object.
SCORABLE_PROMPTS = sorted({v["prompt"] for v in TO_OURS.values() if v["prompt"]})


def download(force: bool = False) -> Path:
    """Fetch and extract the dataset. ~390 MB."""
    DATA.mkdir(parents=True, exist_ok=True)
    marker = DATA / ".extracted"
    if marker.exists() and not force:
        print(f"already present at {DATA}")
        return DATA
    zp = DATA / "homeobjects-3K.zip"
    if not zp.exists() or force:
        print(f"downloading {URL}\n  -> {zp}  (~390 MB)")
        with urllib.request.urlopen(URL) as r, open(zp, "wb") as f:
            shutil.copyfileobj(r, f)
    print("extracting ...")
    with zipfile.ZipFile(zp) as z:
        z.extractall(DATA)
    marker.write_text("ok")
    zp.unlink()
    print(f"done: {DATA}")
    return DATA


def _find_split(split: str) -> tuple[Path, Path]:
    """Locate images/ and labels/ for a split, tolerating an extra nesting level."""
    candidates = [DATA]
    if DATA.exists():
        candidates += [p for p in DATA.iterdir() if p.is_dir()]
    for root in candidates:
        img = root / "images" / split
        lbl = root / "labels" / split
        if img.is_dir():
            return img, lbl
    raise FileNotFoundError(
        f"could not find images/{split} under {DATA}. "
        f"Run: python -m poc.datasets.homeobjects --download")


def load(split: str = "val", limit: int | None = None) -> list[dict]:
    """Return [{image: Path, boxes: [{cls, prompt, box_xyxy_norm}]}].

    YOLO label format is `cls cx cy w h`, all normalised 0-1. We keep it normalised
    here and scale to pixels at use time, because we resize images in stage 1.
    """
    img_dir, lbl_dir = _find_split(split)
    items = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lp = lbl_dir / f"{ip.stem}.txt"
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
                if cid >= len(CLASSES):
                    continue
                name = CLASSES[cid]
                boxes.append(dict(
                    cls=name,
                    prompt=TO_OURS[name]["prompt"],
                    box_norm=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                ))
        items.append(dict(image=ip, boxes=boxes))
        if limit and len(items) >= limit:
            break
    return items


def summary(split: str = "val") -> dict:
    items = load(split)
    counts: dict[str, int] = {}
    for it in items:
        for b in it["boxes"]:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    with_door = sum(1 for it in items if any(b["cls"] == "door" for b in it["boxes"]))
    return dict(split=split, images=len(items),
                boxes=sum(len(i["boxes"]) for i in items),
                per_class=dict(sorted(counts.items(), key=lambda x: -x[1])),
                images_with_a_door=with_door)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HomeObjects-3K helper")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    if a.download:
        download(force=a.force)
    print(json.dumps(summary(a.split), indent=2))
    print(f"\nScorable prompts ({len(SCORABLE_PROMPTS)} of our 26): {SCORABLE_PROMPTS}")
    print("\nAnything we detect outside that list CANNOT be scored on this dataset —")
    print("no label does not mean no object.")
