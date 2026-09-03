"""
FILE PURPOSE
    Run a detector (and optionally depth) over HomeObjects-3K and score DETECTION against
    the dataset's ground-truth boxes. This is the first attempt that produces real numbers.

WHY IT MATTERS
    It closes the detection half of gap A1 for free. No hand-labelling: the dataset already
    has boxes for 12 classes, including `wardrobe` (hard, core removal item) and `door`
    (our scale anchor).

WHAT IT PROVES, AND WHAT IT DOES NOT
    PROVES     which detector finds our furniture, at what precision/recall, and how
               parameter changes affect that. Also whether doors are found reliably enough
               for the scale anchor to work.
    DOES NOT   say anything about volume. The dataset has no depth, no dimensions, no
               camera intrinsics. Depth here is diagnostic only - you can look at it, you
               cannot score it. Gap A1 stays open for measurement.

HOW SCORING WORKS
    A prediction counts as correct (a true positive) if it overlaps a ground-truth box of
    the same class by more than --iou. Each ground-truth box can only be matched once.
      precision = of what we predicted, how much was real
      recall    = of what was there, how much we found
    Only the 8 prompts the dataset can confirm are scored. Detections outside that set are
    counted separately as `unscorable` - absence of a label does not mean absence of object.

USAGE
    python -m poc.datasets.homeobjects --download          # once, ~390MB
    python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50
    python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50 --depth moge2
    python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50 \
        --box-th 0.20 --iou 0.5           # sweep a parameter
    python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30

HOW TO RUN
    python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50
        Score one detector against the dataset's ground-truth boxes.
    python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30
        Sweep box_th and watch precision fall as recall rises.
    python -m poc.runner.run_dataset_eval --detector grounding_dino --depth moge2 --limit 20
        Add a depth sanity check (diagnostic only - this dataset cannot score depth).
    Add --save to write JSON into poc/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC.parent))

from poc.models import registry                                # noqa: E402
from poc.datasets import homeobjects as HO                      # noqa: E402


def iou(a, b) -> float:
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


def score_image(preds, truths, iou_th: float):
    """Greedy one-to-one matching, highest-confidence prediction first."""
    scorable = set(HO.SCORABLE_PROMPTS)
    preds = sorted(preds, key=lambda d: -d.score)
    used = set()
    tp = fp = 0
    unscorable = 0
    per_class = {}
    for p in preds:
        if p.label not in scorable:
            unscorable += 1                 # cannot be judged on this dataset
            continue
        best, best_i = None, 0.0
        for i, t in enumerate(truths):
            if i in used or t["prompt"] != p.label:
                continue
            v = iou(p.box, t["box_px"])
            if v > best_i:
                best, best_i = i, v
        d = per_class.setdefault(p.label, dict(tp=0, fp=0, gt=0))
        if best is not None and best_i >= iou_th:
            used.add(best)
            tp += 1
            d["tp"] += 1
        else:
            fp += 1
            d["fp"] += 1
    n_gt = 0
    for t in truths:
        if t["prompt"] in scorable:
            n_gt += 1
            per_class.setdefault(t["prompt"], dict(tp=0, fp=0, gt=0))["gt"] += 1
    return dict(tp=tp, fp=fp, gt=n_gt, fn=n_gt - tp,
                unscorable=unscorable, per_class=per_class)


def run(detector: str, limit: int, split: str, iou_th: float,
        box_th: float, txt_th: float, depth_key: str | None,
        allow_nc: bool, quiet: bool = False):
    blocked = registry.check_licences([detector, depth_key], allow_nc)
    if blocked:
        print(f"REFUSED — cannot ship: {blocked}. Use --allow-noncommercial to benchmark.")
        return None

    import cv2
    from poc.runner.pipeline import resize_long

    det = registry.build(detector, box_threshold=box_th, text_threshold=txt_th) \
        if detector in ("grounding_dino",) else registry.build(detector)
    dep = registry.build(depth_key) if depth_key else None

    items = HO.load(split, limit=limit)
    prompts = list(json.loads((POC / "detect_vocab.json").read_text())["prompts"].keys())

    agg = dict(tp=0, fp=0, gt=0, fn=0, unscorable=0)
    per_class: dict[str, dict] = {}
    depth_notes = []
    t0 = time.time()

    for n, it in enumerate(items, 1):
        img = cv2.imread(str(it["image"]))
        if img is None:
            continue
        img = resize_long(img, 1024)
        H, W = img.shape[:2]
        truths = []
        for b in it["boxes"]:
            if not b["prompt"]:
                continue
            x0, y0, x1, y1 = b["box_norm"]
            truths.append(dict(prompt=b["prompt"], cls=b["cls"],
                               box_px=[x0 * W, y0 * H, x1 * W, y1 * H]))

        preds = det.detect(img, prompts)
        s = score_image(preds, truths, iou_th)
        for k in ("tp", "fp", "gt", "fn", "unscorable"):
            agg[k] += s[k]
        for c, d in s["per_class"].items():
            pc = per_class.setdefault(c, dict(tp=0, fp=0, gt=0))
            for k in ("tp", "fp", "gt"):
                pc[k] += d[k]

        # Depth is diagnostic only — nothing here can score it.
        if dep is not None and n <= 3:
            g = dep.infer(img)
            v = g.points[..., g.vert_axis][g.mask.astype(bool)]
            depth_notes.append(dict(image=it["image"].name,
                                    depth_min_m=round(float(g.depth[g.mask.astype(bool)].min()), 2),
                                    depth_max_m=round(float(g.depth[g.mask.astype(bool)].max()), 2),
                                    vertical_span_m=round(float(v.max() - v.min()), 2),
                                    valid_pct=round(float(g.mask.mean()) * 100, 1)))
        if not quiet and n % 10 == 0:
            print(f"  {n}/{len(items)}", end="\r")

    elapsed = time.time() - t0
    prec = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) else 0.0
    rec = agg["tp"] / agg["gt"] if agg["gt"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(detector=detector, depth=depth_key, split=split, images=len(items),
                iou_th=iou_th, box_th=box_th, txt_th=txt_th,
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                counts=agg, per_class=per_class, depth_notes=depth_notes,
                seconds=round(elapsed, 1),
                sec_per_image=round(elapsed / max(len(items), 1), 2))


def report(r: dict):
    print(f"\n{'='*72}")
    print(f"  {r['detector']}" + (f" + {r['depth']}" if r["depth"] else "")
          + f"   {r['images']} images, IoU>={r['iou_th']}")
    print(f"  box_th={r['box_th']}  txt_th={r['txt_th']}")
    print(f"{'='*72}")
    c = r["counts"]
    print(f"  precision {r['precision']:.3f}   recall {r['recall']:.3f}   F1 {r['f1']:.3f}")
    print(f"  tp {c['tp']}  fp {c['fp']}  fn {c['fn']}  (ground truth {c['gt']})")
    print(f"  detections we cannot score on this dataset: {c['unscorable']}")
    print(f"  {r['sec_per_image']}s per image  ({r['seconds']}s total)")

    print(f"\n  {'class':16} {'recall':>8} {'prec':>8} {'gt':>5}   <- per class")
    for cls, d in sorted(r["per_class"].items(), key=lambda x: -x[1]["gt"]):
        pr = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        rc = d["tp"] / d["gt"] if d["gt"] else 0.0
        flag = ""
        if cls == "door":
            flag = "  <- SCALE ANCHOR: low recall here breaks stage 4"
        elif cls == "wardrobe" and rc < 0.5:
            flag = "  <- core removal item, poorly found"
        print(f"  {cls:16} {rc:>8.3f} {pr:>8.3f} {d['gt']:>5}{flag}")

    if r["depth_notes"]:
        print(f"\n  depth sanity check (DIAGNOSTIC ONLY — nothing here can score it):")
        for d in r["depth_notes"]:
            print(f"    {d['image'][:28]:28} range {d['depth_min_m']}-{d['depth_max_m']}m  "
                  f"vertical span {d['vertical_span_m']}m  valid {d['valid_pct']}%")
        print("    A plausible room: depth 1-8m, vertical span 2-3m, valid >80%.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a detector on HomeObjects-3K.")
    ap.add_argument("--detector", default="grounding_dino", choices=registry.keys("detector"))
    ap.add_argument("--depth", default=None, choices=registry.keys("depth"),
                    help="optional, diagnostic only — this dataset cannot score depth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a match to count")
    ap.add_argument("--box-th", type=float, default=0.30)
    ap.add_argument("--txt-th", type=float, default=0.25)
    ap.add_argument("--allow-noncommercial", action="store_true")
    ap.add_argument("--compare-thresholds", action="store_true",
                    help="sweep box_th to see the precision/recall trade directly")
    ap.add_argument("--save", action="store_true", help="write JSON to poc/results/")
    a = ap.parse_args(argv)

    if a.compare_thresholds:
        print("Sweeping box_th — watch precision fall as recall rises.\n")
        print(f"  {'box_th':>7} {'precision':>10} {'recall':>8} {'F1':>7} {'fp':>6} {'fn':>6}")
        rows = []
        for th in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            r = run(a.detector, a.limit, a.split, a.iou, th, a.txt_th,
                    None, a.allow_noncommercial, quiet=True)
            if not r:
                return 2
            rows.append(r)
            print(f"  {th:>7.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                  f"{r['f1']:>7.3f} {r['counts']['fp']:>6} {r['counts']['fn']:>6}")
        best = max(rows, key=lambda x: x["f1"])
        print(f"\n  best F1 at box_th={best['box_th']}  (F1 {best['f1']})")
        print("  But for a SURVEY prefer recall: a missed wardrobe is a 100% error on that")
        print("  item, while a false positive is deleted by the reviewer in two seconds.")
        return 0

    r = run(a.detector, a.limit, a.split, a.iou, a.box_th, a.txt_th,
            a.depth, a.allow_noncommercial)
    if not r:
        return 2
    report(r)
    if a.save:
        out = POC / "results"
        out.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        f = out / f"dataseteval__{a.detector}__{stamp}.json"
        r["dataset"] = "HomeObjects-3K"
        r["scores_detection_only"] = True
        f.write_text(json.dumps(r, indent=2))
        print(f"\nwrote {f.relative_to(POC.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
