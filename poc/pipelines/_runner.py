"""
FILE PURPOSE
    Turns a pipeline's config.py into a run of the shared orchestrator. Every
    pipeline's run.py is three lines because this holds the argument plumbing.

WHY IT EXISTS
    A pipeline folder should say WHAT to run, not HOW to run it. This reads the
    config, points input and output at the pipeline's own folders, applies any
    command-line override, and hands off to poc.runner.run_combination — the same
    code path the general-purpose CLI uses, so a pipeline cannot behave differently
    from the tool it is supposed to be a preset of.

USED BY  poc/pipelines/<name>/run.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_VID = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def folder_of(cfg) -> Path:
    return Path(cfg.__file__).resolve().parent


def candidates(in_dir: Path) -> list[str]:
    """What could be run: images, videos, and folders of stills (one room each)."""
    if not in_dir.is_dir():
        return []
    return sorted(q.name for q in in_dir.iterdir()
                  if (q.is_dir() and q.name != ".ipynb_checkpoints")
                  or q.suffix.lower() in IMG_VID)


def run_from_config(cfg, argv=None) -> int:
    here = folder_of(cfg)
    in_dir, res_dir = here / "data" / "input", here / "results"

    ap = argparse.ArgumentParser(
        prog=f"poc.pipelines.{here.name}.run",
        description=f"{getattr(cfg, 'QUESTION', here.name)}\n\n"
                    f"input  {in_dir}\nresults {res_dir}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="file or folder under data/input. Omitted: used only if "
                         "exactly one thing is there.")
    ap.add_argument("--room", default=None, help="room_id in ground_truth.csv, to score")
    # Overrides. Left as None so the config wins unless you say otherwise, and so
    # the config stays the single record of what this pipeline is.
    ap.add_argument("--box-th", type=float, default=None)
    ap.add_argument("--txt-th", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-noncommercial", action="store_true")
    a = ap.parse_args(argv)

    chosen = a.input
    if chosen is None:
        found = candidates(in_dir)
        if len(found) == 1:
            chosen = found[0]
            print(f"using the only input in data/input: {chosen}")
        elif not found:
            print(f"nothing to run. Put a photo, a video, or a folder of stills of one "
                  f"room in:\n    {in_dir}")
            return 1
        else:
            print(f"{len(found)} inputs in data/input — name one with --input:")
            for f in found:
                print(f"    --input {f}")
            print("\nOne run per room: each invocation reloads the models and pays the "
                  "~370 s\nGrounding DINO warm-up, so this is not a batching shortcut.")
            return 1

    def pick(cli, cfg_name, default=None):
        """Command line beats config beats default."""
        if cli is not None:
            return cli
        return getattr(cfg, cfg_name, default)

    args = ["--input", chosen,
            "--input-dir", str(in_dir),
            "--results-dir", str(res_dir),
            "--detector", cfg.DETECTOR,
            "--depth", cfg.DEPTH,
            "--dedup", getattr(cfg, "DEDUP", "max"),
            "--max-frames", str(pick(a.max_frames, "MAX_FRAMES", 16))]

    if getattr(cfg, "SEGMENTER", None):
        args += ["--segmenter", cfg.SEGMENTER]
    args += ["--classifier", getattr(cfg, "CLASSIFIER", None) or "none"]

    box = pick(a.box_th, "BOX_TH")
    txt = pick(a.txt_th, "TEXT_TH")
    if box is not None:
        args += ["--box-th", str(box)]
    if txt is not None:
        args += ["--txt-th", str(txt)]
    if getattr(cfg, "NO_ANCHOR", False):
        args += ["--no-anchor"]
    if a.room:
        args += ["--room", a.room]
    if a.no_vis:
        args += ["--no-vis"]
    if a.tag:
        args += ["--tag", a.tag]
    if a.allow_noncommercial:
        args += ["--allow-noncommercial"]

    print(f"=== pipeline {here.name} ===")
    print(f"    {getattr(cfg, 'QUESTION', '')}")
    print(f"    input   {in_dir}")
    print(f"    results {res_dir}\n")

    from poc.runner import run_combination
    return run_combination.main(args)
