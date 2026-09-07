"""
FILE PURPOSE
    Run ONE model combination over one input file and write a single result JSON.
    This is the unit of the whole comparison — every row in the final ranking table
    comes from one invocation of this file.

WHY IT'S A CLI RATHER THAN A NOTEBOOK
    A combination has to be reproducible and scriptable. The sweep runs this many
    times with different flags; a notebook cannot be driven that way.

USAGE
    python -m poc.runner.run_combination --detector grounding_dino --depth moge2 \
        --classifier claude_sonnet5 --input data/input/bedroom.mp4 --room BED01

    # ablation: the same thing with the scale anchor switched off
    python -m poc.runner.run_combination --detector grounding_dino --depth moge2 \
        --no-anchor --input data/input/bedroom.mp4 --room BED01

    # licence-blocked models require an explicit opt-in
    python -m poc.runner.run_combination --detector yolo_world --allow-noncommercial ...

OUTPUT
    poc/results/<combo_id>__<input>__<timestamp>.json holding the full config, the
    inventories from both methods, the volumes, timings, cost, and — if
    ground_truth.csv covers the room — the scores.

LICENCE GATE
    Refuses any combination containing a non-commercial model unless
    --allow-noncommercial is passed. Results from such a run are permanently tagged
    shippable=false so they can never be mistaken for a viable configuration.

HOW TO RUN
    python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
        --detector grounding_dino --depth moge2 --classifier claude_sonnet5
        One combination on one room video. Writes one JSON to poc/results/.
    Add --no-anchor for the key experiment (same config, scale anchor off).
    Add --segmenter sam2 to upgrade boxes to masks.
    Add --classifier none to skip Method A entirely.
    See every option:  python -m poc.runner.run_combination --help
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../poc
sys.path.insert(0, str(ROOT.parent))                # so `poc.` imports resolve

from poc.models import registry                                     # noqa: E402
from poc.runner import pipeline as P                                 # noqa: E402


def load_tables():
    cube = json.loads((ROOT / "cube_table.json").read_text())
    vocab = json.loads((ROOT / "detect_vocab.json").read_text())["prompts"]
    return cube, cube["classes"], vocab


def load_truth(room: str | None):
    """Ground truth counts for one room, or None if unavailable."""
    p = ROOT / "ground_truth.csv"
    if not p.exists() or not room:
        return None, None
    rows = [r for r in csv.DictReader(l for l in p.open() if not l.startswith("#"))
            if r["room_id"] == room]
    if not rows:
        return None, None
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["true_class"]] = counts.get(r["true_class"], 0) + int(r["qty"])
    dims = {r["true_class"]: (float(r["width_m"]), float(r["depth_m"]), float(r["height_m"]))
            for r in rows}
    return counts, dims


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one model combination.")
    ap.add_argument("--input", required=True, help="file under poc/data/input/")
    ap.add_argument("--detector", default="grounding_dino", choices=registry.keys("detector"))
    ap.add_argument("--depth", default="moge2", choices=registry.keys("depth"))
    ap.add_argument("--segmenter", default=None,
                    choices=registry.keys("segmenter"), help="box -> mask refinement")
    ap.add_argument("--classifier", default="claude_sonnet5",
                    choices=registry.keys("classifier") + ["none"])
    ap.add_argument("--no-anchor", action="store_true", help="disable the scale anchor (ablation)")
    ap.add_argument("--room", default=None, help="room_id in ground_truth.csv, for scoring")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--vlm-frames", type=int, default=6)
    ap.add_argument("--dedup", default="max", choices=["max", "median"])
    ap.add_argument("--allow-noncommercial", action="store_true")
    ap.add_argument("--tag", default="", help="free-text label for this run")
    a = ap.parse_args(argv)

    classifier_key = None if a.classifier == "none" else a.classifier
    selected = [a.detector, a.depth, a.segmenter, classifier_key]

    # ---- licence gate ----------------------------------------------------
    blocked = registry.check_licences(selected, a.allow_noncommercial)
    if blocked:
        print("REFUSED — this combination contains models that cannot ship:")
        for k in blocked:
            i = registry.info(k)
            print(f"  {k}: {i.licence}")
            if i.warn:
                print(f"      {i.warn}")
        print("\nPass --allow-noncommercial to run it anyway for comparison purposes.")
        print("The result will be tagged shippable=false.")
        return 2

    shippable = not [k for k in selected if k and not registry.info(k).commercial_ok]
    for k in [x for x in selected if x]:
        w = registry.info(k).warn
        if w:
            print(f"  ! {k}: {w}")

    cube, classes, vocab = load_tables()
    class_names = sorted(classes)
    prompts = list(vocab.keys())

    src = ROOT / "data" / "input" / a.input if not Path(a.input).is_absolute() else Path(a.input)
    if not src.exists():
        src = Path(a.input)
    if not src.exists():
        print(f"input not found: {a.input}")
        return 1

    combo_id = "__".join([a.detector, a.depth, a.segmenter or "nomask",
                          classifier_key or "nocls",
                          "noanchor" if a.no_anchor else "anchor"])
    print(f"\n=== {combo_id} ===")
    print(f"input {src.name}   shippable={shippable}")

    timings: dict[str, float] = {}

    # ---- stage 1 ---------------------------------------------------------
    t0 = time.time()
    sampled, frames = P.load_keyframes(src, max_frames=a.max_frames)
    timings["ingest_s"] = round(time.time() - t0, 2)
    print(f"stage 1  {len(frames)}/{len(sampled)} frames kept  ({timings['ingest_s']}s)")
    if not frames:
        print("no usable frames")
        return 1

    det = registry.build(a.detector)
    dep = registry.build(a.depth)
    seg = registry.build(a.segmenter) if a.segmenter else None

    # ---- stage 2 detection ----------------------------------------------
    t0 = time.time()
    for f in frames:
        f["dets"] = det.detect(f["img"], prompts)
        if seg is not None:
            boxes = [d.box for d in f["dets"]]
            for d, m in zip(f["dets"], seg.refine(f["img"], boxes)):
                d.mask = m
    timings["detect_s"] = round(time.time() - t0, 2)
    n_det = sum(len(f["dets"]) for f in frames)
    unreachable = det.unreachable(prompts) if hasattr(det, "unreachable") else []
    print(f"stage 2  {n_det} detections  ({timings['detect_s']}s)")
    if unreachable:
        print(f"         closed vocabulary — cannot even attempt: {len(unreachable)} prompts")

    # ---- stage 3 depth ---------------------------------------------------
    t0 = time.time()
    for f in frames:
        f["depth"] = dep.infer(f["img"])
    timings["depth_s"] = round(time.time() - t0, 2)
    print(f"stage 3  depth done  ({timings['depth_s']}s)")

    # ---- stage 4 scale anchor -------------------------------------------
    factors, whys = [], []
    for f in frames:
        fac, why = P.door_scale(f["dets"], f["depth"])
        whys.append(why)
        if fac:
            factors.append(fac)
    import numpy as np
    if a.no_anchor or not factors:
        scale = 1.0
        scale_src = "anchor disabled" if a.no_anchor else "no door found"
    else:
        scale = float(np.median(factors))
        scale_src = f"door anchor, median of {len(factors)} frames"
    print(f"stage 4  SCALE={scale:.4f}  [{scale_src}]")

    # ---- stage 5 Method B -----------------------------------------------
    t0 = time.time()
    rows = []
    for f in frames:
        for d in f["dets"]:
            # Exact match, not `"door" in d.label`. Labels are normalised to a
            # single vocabulary prompt now (see normalise_label), and the old
            # substring test matched the raw span 'sofa wardrobe door chair',
            # throwing away a real sofa as if it were the anchor. 'door' is the
            # only door-like prompt in the 26, so equality is sufficient.
            if d.label == "door":
                continue
            e = P.extent(f["depth"], d, scale=scale)
            if not e:
                continue
            # resolve_dims handles the single-view depth limit: a camera never sees
            # the back of a wardrobe, so for flat-fronted objects the depth extent is
            # unobservable and a class prior is substituted. See gap B8.
            dims, cls, dist, depth_src = P.resolve_dims(e, classes, vocab.get(d.label) or None)
            rows.append(dict(t=f["t"], det_label=d.label, score=round(d.score, 3),
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in dims.items()},
                             mapped_class=cls, map_dist=round(dist, 3),
                             depth_source=depth_src))
    timings["measure_s"] = round(time.time() - t0, 2)

    # Label quality. An open-vocabulary detector returns matched text, not a
    # class id, so a box can arrive unlabelled ('') or fused from several
    # prompts. normalise_label() repairs what it can; what it cannot repair
    # would otherwise widen the size-class search from 2-3 candidates to all 42
    # and cost ~18 points of accuracy with nothing in the output to show it.
    # So it is counted here and reported in the JSON.
    all_dets = [d for f in frames for d in f["dets"]]
    reasons: dict[str, int] = {}
    for d in all_dets:
        r = getattr(d, "label_reason", "") or "clean"
        reasons[r] = reasons.get(r, 0) + 1
    lab_missing = sum(1 for d in all_dets if not d.label)
    lab_fixed = reasons.get("narrowed", 0)
    if all_dets:
        n = len(all_dets)
        print(f"         labels: " + ", ".join(f"{v} {k}" for k, v in sorted(reasons.items()))
              + f" (of {n})")
        # The two failures need OPPOSITE corrections, which is why the reason is
        # carried all the way here instead of just a count of bad labels.
        if reasons.get("empty", 0) / n > 0.25:
            print(f"         ! {reasons['empty']}/{n} boxes were named by nothing: "
                  "text_threshold is too HIGH, no token cleared it.")
            print("           LOWER it, 0.25 -> 0.15, and re-run. LEARN.md Lesson 1.")
        if reasons.get("ambiguous", 0) / n > 0.25:
            print(f"         ! {reasons['ambiguous']}/{n} spans fused several prompts: "
                  "text_threshold is too LOW.")
            print("           RAISE it, 0.25 -> 0.35. These boxes are measured but "
                  "unnamed — we refuse to guess a class (see normalise_label).")
        if reasons.get("unmatched", 0):
            print(f"         ! {reasons['unmatched']} labels matched no prompt at all — "
                  "check detect_vocab.json against the model's vocabulary.")

    masked = sum(1 for r in rows if r.get("used_mask"))
    prior = sum(1 for r in rows if r.get("depth_source") == "class_prior")
    print(f"stage 5  {len(rows)} objects measured, {masked} using masks")
    if rows:
        pct = prior / len(rows) * 100
        print(f"         depth unobservable on {prior}/{len(rows)} ({pct:.0f}%) "
              f"— class prior used")
        if pct > 60:
            print("         ! Most depths came from the cube table, not from geometry.")
            print("           Method B is leaning on Method A's data. Weigh the comparison "
                  "accordingly.")

    # ---- stage 6 Method A -----------------------------------------------
    per_frame, cost, usage_tot = [], 0.0, dict(input_tokens=0, output_tokens=0)
    if classifier_key:
        t0 = time.time()
        cls_model = registry.build(classifier_key)
        for f in frames[: a.vlm_frames]:
            items, u = cls_model.classify(f["img"], class_names)
            per_frame.append(dict(t=f["t"], room_type=u.get("room_type", "?"),
                                  items=[dict(size_class=i.size_class, count=i.count,
                                              confidence=i.confidence) for i in items]))
            cost += u["cost_usd"]
            usage_tot["input_tokens"] += u["input_tokens"]
            usage_tot["output_tokens"] += u["output_tokens"]
        timings["classify_s"] = round(time.time() - t0, 2)
        print(f"stage 6  {len(per_frame)} frames classified  "
              f"(${cost:.3f}, {timings['classify_s']}s)")

    # ---- stages 7-8 -----------------------------------------------------
    inv_recognised = P.dedup_counts(per_frame, a.dedup)
    inv_measured = P.dedup_measured(rows, a.dedup)
    vols = dict(
        recognised_m3=round(P.vol_from_inventory(inv_recognised, classes), 3),
        measured_class_m3=round(P.vol_from_inventory(inv_measured, classes), 3),
        measured_raw_m3=round(sum(r["bbox_m3"] for r in rows), 3),
    )
    print(f"stage 8  A={vols['recognised_m3']} m3   "
          f"B(class)={vols['measured_class_m3']} m3   B(raw)={vols['measured_raw_m3']} m3")

    # ---- stage 9 scoring -------------------------------------------------
    truth, truth_dims = load_truth(a.room)
    scores = {}
    if truth:
        if inv_recognised:
            scores["method_a"] = P.score_inventory(inv_recognised, truth, classes)
        if inv_measured:
            scores["method_b"] = P.score_inventory(inv_measured, truth, classes)
        for k, v in scores.items():
            print(f"stage 9  {k}: bias {v['vol_bias_pct']}%  abs {v['vol_abs_err_pct']}%  "
                  f"P {v['precision']} R {v['recall']}")
    else:
        print("stage 9  no ground truth for this room — not scored")

    # ---- write the result ------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "combo_id": combo_id,
        "tag": a.tag,
        "timestamp": stamp,
        "shippable": shippable,
        "input": src.name,
        "room": a.room,
        "config": {
            "detector": a.detector, "depth": a.depth, "segmenter": a.segmenter,
            "classifier": classifier_key, "scale_anchor": not a.no_anchor,
            "dedup": a.dedup, "max_frames": a.max_frames, "vlm_frames": a.vlm_frames,
        },
        "licences": {k: registry.info(k).licence for k in selected if k},
        "frames": {"sampled": len(sampled), "kept": len(frames)},
        "scale": {"factor": round(scale, 4), "source": scale_src,
                  "per_frame": [round(x, 4) for x in factors],
                  "reasons": whys[:3]},
        "detection": {"count": n_det, "unreachable_prompts": unreachable,
                      "masked_measurements": masked},
        "depth_observability": {
            "measured": len(rows),
            "class_prior": prior,
            "class_prior_pct": round(prior / len(rows) * 100, 1) if rows else None,
            "note": "class_prior means the object's depth was never visible to the "
                    "camera, so the cube table supplied it. A high fraction means "
                    "Method B is not measuring independently.",
        },
        "label_quality": {
            "detections": len(all_dets),
            "by_reason": reasons,
            "normalised": lab_fixed,
            "unnamed": lab_missing,
            "note": "Open-vocabulary detectors return matched text, not class ids. "
                    "'normalised' had a multi-prompt span reduced to one class; "
                    "'unnamed' cleared the box threshold but no token cleared "
                    "text_threshold, so the size-class search could not be "
                    "restricted. Both are tuning signals, not errors.",
        },
        "inventory": {"method_a": inv_recognised, "method_b": inv_measured},
        "volumes": vols,
        "measurements": rows,
        "scores": scores,
        "cost_usd": round(cost, 5),
        "usage": usage_tot,
        "timings": timings,
    }
    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    fn = res_dir / f"{combo_id}__{src.stem}__{stamp}.json"
    fn.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fn.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
