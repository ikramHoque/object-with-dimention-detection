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
import shlex
from pathlib import Path

# What counts as an input lives in ONE place, poc/runner/pipeline.py, and so does
# the rule that a sub-folder is a room. This module only chooses WHICH rooms to run.
from poc.runner.pipeline import discover_rooms


def folder_of(cfg) -> Path:
    return Path(cfg.__file__).resolve().parent


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
    # One image = one room. Several rooms in one process is much cheaper than
    # several processes, because the ~370 s detector warm-up is paid once.
    ap.add_argument("--all", action="store_true",
                    help="run EVERY input in data/input, each as its own room. "
                         "Models load once and are reused, so room 2 onwards "
                         "costs seconds.")
    ap.add_argument("--room", default=None, help="room_id in ground_truth.csv, to score")
    # Overrides. Left as None so the config wins unless you say otherwise, and so
    # the config stays the single record of what this pipeline is.
    ap.add_argument("--box-th", type=float, default=None)
    ap.add_argument("--txt-th", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    # Most combinations specify a Claude classifier, which needs ANTHROPIC_API_KEY and
    # costs money. `--classifier none` runs Method B (measurement) alone, so a pipeline
    # can be tried without a key. It changes what is measured, so it is an override you
    # ask for rather than a default.
    ap.add_argument("--classifier", default=None,
                    help="override the config: a classifier key, or 'none' for "
                         "geometry only (no API key needed)")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-noncommercial", action="store_true")
    a = ap.parse_args(argv)

    rooms = discover_rooms(in_dir)

    # An explicit --input is honoured WITHOUT consulting the listing: it may be one
    # image inside a room, an absolute path outside data/input, or a name the
    # extension filter does not recognise. run_combination reports it clearly if it
    # does not exist. Only the cases that have to GUESS need the listing.
    if a.input is not None:
        todo = [a.input]
    elif not rooms:
        print("nothing to run. A ROOM is a folder of photographs, so make one:\n")
        print(f"    mkdir -p {in_dir / 'bedroom'}")
        print(f"    cp your_photos/*.jpg {in_dir / 'bedroom'}/\n")
        print("One folder per room; the folder name becomes the room name. A single")
        print("loose image also works, as a one-photo room named after the file.")
        return 1
    elif a.all:
        todo = [r["path"].name for r in rooms]
    elif len(rooms) == 1:
        todo = [rooms[0]["path"].name]
        print(f"using the only room in data/input: {rooms[0]['name']}")
    else:
        folders = [r for r in rooms if r["kind"] == "folder"]
        print(f"{len(rooms)} rooms in data/input"
              f"{f' ({len(folders)} as folders)' if folders else ''}:\n")
        for r in rooms:
            note = (f"{r['n_images']} photo(s)" if r["kind"] == "folder"
                    else f"a single {r['kind']}")
            print(f"    {r['name']:<32} {note}")
        print("\n  every room, in one process (the models load once):")
        print(f"      python -m poc.pipelines.{here.name}.run --all")
        print("\n  one room:")
        # shlex.quote, because real names contain spaces and brackets
        # ("living room (front)") and an unquoted hint cannot be pasted.
        print(f"      --input {shlex.quote(rooms[0]['path'].name)}")
        print("\n  one photograph out of a room:")
        ex = next((r for r in rooms if r["kind"] == "folder"), None)
        if ex:
            first = sorted(q.name for q in ex["path"].iterdir()
                           if not q.name.startswith("."))[0]
            print(f"      --input {shlex.quote(ex['name'] + '/' + first)}")
        else:
            print("      (no room folders yet — make one to try this)")
        return 1

    def pick(cli, cfg_name, default=None):
        """Command line beats config beats default."""
        if cli is not None:
            return cli
        return getattr(cfg, cfg_name, default)

    args = ["--input-dir", str(in_dir),
            "--results-dir", str(res_dir),
            "--detector", cfg.DETECTOR,
            "--depth", cfg.DEPTH,
            "--dedup", getattr(cfg, "DEDUP", "max"),
            "--max-frames", str(pick(a.max_frames, "MAX_FRAMES", 16))]

    if getattr(cfg, "SEGMENTER", None):
        args += ["--segmenter", cfg.SEGMENTER]
    args += ["--classifier",
             a.classifier or getattr(cfg, "CLASSIFIER", None) or "none"]

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
    print(f"    results {res_dir}")
    if len(todo) > 1:
        print(f"    rooms   {len(todo)}: {', '.join(todo)}")
    print()

    from poc.runner import run_combination

    # --room names a row in ground_truth.csv, so it identifies ONE room and cannot
    # be applied to a batch. Scoring a batch needs a room id per input, which is a
    # mapping we do not have; better to refuse than to score every room against one
    # row and report a meaningless error.
    if a.room and len(todo) > 1:
        print("--room scores one room against ground_truth.csv and cannot describe "
              f"{len(todo)} of them.\nRun each room separately with its own --room.")
        return 1

    failed = []
    for i, one in enumerate(todo, 1):
        if len(todo) > 1:
            print(f"\n{'=' * 70}\nroom {i}/{len(todo)}  {one}\n{'=' * 70}")
        rc = run_combination.main(["--input", one, *args])
        if rc != 0:
            failed.append(one)

    if len(todo) > 1:
        ok = len(todo) - len(failed)
        print(f"\n{'=' * 70}")
        print(f"{ok}/{len(todo)} rooms completed. Results in {res_dir}")
        if failed:
            print(f"failed: {', '.join(failed)}")
        print("Each room wrote its own JSON, annotated frame and items.csv — one")
        print("inventory per room, never merged, because they are different rooms.")
    return 1 if failed else 0
