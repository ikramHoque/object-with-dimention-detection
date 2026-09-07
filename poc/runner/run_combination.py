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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../poc
sys.path.insert(0, str(ROOT.parent))                # so `poc.` imports resolve

from poc.models import registry                                     # noqa: E402
from poc.runner import pipeline as P                                 # noqa: E402
from poc.runner import report as R                                  # noqa: E402


def load_tables():
    cube = json.loads((ROOT / "cube_table.json").read_text())
    vocab = json.loads((ROOT / "detect_vocab.json").read_text())["prompts"]
    return cube, cube["classes"], vocab


def safe_name(s: str) -> str:
    """A room name that is safe in a filename. Real folders are called things like
    "living room (front)", and those characters make results awkward to handle from
    a shell. The readable name is preserved inside the JSON and the CSV."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-") or "room"


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
    # Where input comes from and where output goes. Default to the shared
    # poc/data/input and poc/results; a pipeline folder passes its own, so its
    # photographs and results live beside its config and notebook.
    ap.add_argument("--box-th", type=float, default=None,
                    help="detector confidence threshold (default: the adapter's own)")
    ap.add_argument("--txt-th", type=float, default=None,
                    help="open-vocabulary text threshold, where the model has one")
    ap.add_argument("--input-dir", default=None,
                    help="folder to resolve --input against (default poc/data/input)")
    ap.add_argument("--results-dir", default=None,
                    help="where to write JSON, frames and CSV (default poc/results)")
    ap.add_argument("--no-vis", action="store_true",
                    help="skip the annotated JPEGs and the CSV (JSON only)")
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

    in_dir = Path(a.input_dir).expanduser().resolve() if a.input_dir else ROOT / "data" / "input"
    res_dir = Path(a.results_dir).expanduser().resolve() if a.results_dir else ROOT / "results"

    def disp(q: Path) -> str:
        """Show a short path when we can, an absolute one when we cannot. A
        pipeline folder may sit outside the repo, where relative_to() raises."""
        try:
            return str(q.relative_to(ROOT.parent))
        except ValueError:
            return str(q)

    src = in_dir / a.input if not Path(a.input).is_absolute() else Path(a.input)
    if not src.exists():
        src = Path(a.input)
    if not src.exists():
        print(f"input not found: {a.input}  (looked in {disp(in_dir)})")
        return 1

    # The ROOM this input belongs to. A sub-folder of data/input IS a room and its
    # name is the room name, so `--input lounge/north.jpg` is filed under "lounge",
    # not "north". Every output carries this: the JSON, the CSV, the annotated
    # frame and the filenames.
    room_name = P.room_name_for(src, in_dir)
    # The folder name doubles as the ground_truth.csv room_id, because they are the
    # same thing — the room. --room overrides it for a differently-keyed sheet.
    gt_room = a.room or room_name

    # One timestamp for the whole run, set before anything is written, so the
    # results JSON and the annotated-frame folder always share the same stem.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    combo_id = "__".join([a.detector, a.depth, a.segmenter or "nomask",
                          classifier_key or "nocls",
                          "noanchor" if a.no_anchor else "anchor"])
    print(f"\n=== {combo_id} ===")
    print(f"room  {room_name}")
    print(f"input {src.name}   shippable={shippable}")

    # Thresholds reach the adapter here. Before this they did not: the detector was
    # built with no arguments, so --box-th had nowhere to go and every run silently
    # used the adapter default. registry.build drops what a model cannot take.
    det_kwargs = {}
    if a.box_th is not None:
        det_kwargs["box_threshold"] = a.box_th
    if a.txt_th is not None:
        det_kwargs["text_threshold"] = a.txt_th
    det = registry.build(a.detector, **det_kwargs)
    dep = registry.build(a.depth)
    seg = registry.build(a.segmenter) if a.segmenter else None

    # ---- stages 1-5 ------------------------------------------------------
    # One shared implementation, in pipeline.py, so this and every pipeline
    # notebook run byte-identical arithmetic. See measure_room's docstring for
    # the bug that two copies had already caused.
    M = P.measure_room(src, det, dep, seg, prompts=prompts, classes=classes,
                       vocab=vocab, max_frames=a.max_frames,
                       use_anchor=not a.no_anchor, log=print)
    frames, rows, timings = M["frames"], M["rows"], M["timings"]
    if not frames:
        print("no usable frames")
        return 1
    sampled, scale, scale_src = M["sampled"], M["scale"], M["scale_src"]
    factors, whys, n_det = M["factors"], M["whys"], M["n_det"]
    unreachable, masked, prior = M["unreachable"], M["masked"], M["prior"]
    all_dets, reasons = M["all_dets"], M["reasons"]
    lab_missing, lab_fixed = M["lab_missing"], M["lab_fixed"]

    if unreachable:
        print(f"         closed vocabulary — cannot even attempt: "
              f"{len(unreachable)} prompts")

    # Label quality. An open-vocabulary detector returns matched text, not a class
    # id, so a box can arrive unlabelled ('') or fused from several prompts.
    # normalise_label() repairs what it can; what it cannot repair would otherwise
    # widen the size-class search from 2-3 candidates to all 42.
    if all_dets:
        n = len(all_dets)
        print("         labels: " + ", ".join(f"{v} {k}" for k, v in sorted(reasons.items()))
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

    if rows:
        pct = prior / len(rows) * 100
        print(f"         depth unobservable on {prior}/{len(rows)} ({pct:.0f}%) "
              f"— class prior used")
        if pct > 60:
            print("         ! Most depths came from the cube table, not from geometry.")
            print("           Method B is leaning on Method A's data. Weigh the "
                  "comparison accordingly.")

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

    # ---- the human-readable half -----------------------------------------
    # The JSON above is for compare.py. A person deciding whether to trust these
    # numbers needs to SEE the boxes and read a table, which is what this is.
    # Room first, then the timestamp. The timestamp means a re-run ADDS a report
    # rather than replacing the last one, so you can compare two settings after the
    # fact instead of losing the earlier answer.
    stem = f"{combo_id}__{safe_name(room_name)}__{stamp}"
    vis_dir = res_dir / stem
    if not a.no_vis:
        imgs = R.write_visuals(frames, vis_dir, scale=scale,
                               total_m3=vols["measured_class_m3"])
        csv_path = R.write_csv(rows, inv_measured, classes, vis_dir / "items.csv")
        print(f"\nwrote {len(imgs)} annotated frame(s) to "
              f"{disp(vis_dir)}/")
        print(f"      green measured · amber depth assumed · red unnamed · blue door")
        print(f"wrote {disp(csv_path)}")
    print("\n".join(R.item_lines(rows, inv_measured, classes)))

    # ---- stage 9 scoring -------------------------------------------------
    truth, truth_dims = load_truth(gt_room)
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
        print(f"stage 9  no ground_truth.csv row for room '{gt_room}' — not scored")

    # ---- write the result ------------------------------------------------
    out = {
        "combo_id": combo_id,
        "tag": a.tag,
        "timestamp": stamp,
        "shippable": shippable,
        "input": src.name,
        "room": room_name,
        "room_source": "folder name" if a.room is None else "--room",
        "ground_truth_room": gt_room,
        "config": {
            "detector": a.detector, "depth": a.depth, "segmenter": a.segmenter,
            "classifier": classifier_key, "scale_anchor": not a.no_anchor,
            "dedup": a.dedup, "max_frames": a.max_frames, "vlm_frames": a.vlm_frames,
            "box_th": a.box_th, "txt_th": a.txt_th,
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
    res_dir.mkdir(parents=True, exist_ok=True)
    fn = res_dir / f"{stem}.json"
    fn.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {disp(fn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
