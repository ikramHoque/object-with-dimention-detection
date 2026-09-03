"""
FILE PURPOSE
    Run the curated set of combinations from combinations.json, one after another, and
    leave one result JSON per run for compare.py to rank.

WHY ONE-FACTOR-AT-A-TIME AND NOT THE FULL GRID
    The full cartesian product is 288 runs, roughly 19 hours on an M1, and most cells
    answer no question anyone asked. This sweep starts from a baseline and changes
    exactly ONE thing per run, so a difference in the result has exactly one possible
    cause. 13 runs, 13 answers.

    If two factors turn out to interact, add a targeted pair afterwards. Do not grid
    pre-emptively.

USAGE
    python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01
    python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --only anchor_off,det_sam3
    python -m poc.runner.run_sweep --input bedroom.mp4 --dry-run     # print the plan
    python -m poc.runner.run_sweep --input bedroom.mp4 --skip-unavailable

NOTE
    A failing run does not stop the sweep. Failures are collected and reported at the
    end, because a missing gated checkpoint should not cost you the other twelve runs.

HOW TO RUN
    python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --dry-run
        Print the 14-run plan with licence and install warnings. Runs nothing.
    python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --skip-unavailable
        Execute the plan, skipping runs whose models are not installed.
    python -m poc.runner.run_sweep --input bedroom.mp4 --only anchor_off,det_sam3
        Just those two runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from poc.models import registry                          # noqa: E402
from poc.runner.run_combination import main as run_one    # noqa: E402


def plan(cfg: dict, only: set[str] | None):
    runs = [cfg["baseline"]] + cfg["runs"]
    if only:
        runs = [r for r in runs if r["id"] in only]
    return runs


def models_in(args: list[str]) -> list[str]:
    keys = []
    for flag in ("--detector", "--depth", "--segmenter", "--classifier"):
        if flag in args:
            v = args[args.index(flag) + 1]
            if v != "none":
                keys.append(v)
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the curated combination sweep.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--room", default=None)
    ap.add_argument("--only", default=None, help="comma-separated run ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-unavailable", action="store_true",
                    help="skip runs whose models are not installed")
    a = ap.parse_args(argv)

    cfg = json.loads((ROOT / "combinations.json").read_text())
    only = set(a.only.split(",")) if a.only else None
    runs = plan(cfg, only)

    print(f"\nSWEEP · {len(runs)} runs on {a.input}"
          + (f" (room {a.room})" if a.room else ""))
    print("=" * 78)
    for r in runs:
        keys = models_in(r["args"])
        missing = [k for k in keys if not registry.is_installed(k)]
        nc = [k for k in keys if not registry.info(k).commercial_ok]
        flags = []
        if missing:
            flags.append(f"NOT INSTALLED: {','.join(missing)}")
        if nc:
            flags.append(f"cannot ship: {','.join(nc)}")
        print(f"\n  [{r['id']}]  axis: {r.get('axis','baseline')}")
        print(f"    {r['question']}")
        if flags:
            print(f"    ! {' | '.join(flags)}")

    if a.dry_run:
        print("\n(dry run — nothing executed)")
        return 0

    results, failures, skipped = [], [], []
    for i, r in enumerate(runs, 1):
        keys = models_in(r["args"])
        missing = [k for k in keys if not registry.is_installed(k)]
        if missing and a.skip_unavailable:
            skipped.append((r["id"], f"not installed: {','.join(missing)}"))
            continue
        argv_run = list(r["args"]) + ["--input", a.input, "--tag", r["id"]]
        if a.room:
            argv_run += ["--room", a.room]
        print(f"\n{'='*78}\n[{i}/{len(runs)}] {r['id']}\n{'='*78}")
        t0 = time.time()
        try:
            rc = run_one(argv_run)
            (results if rc == 0 else failures).append((r["id"], rc))
        except Exception as e:                       # keep the sweep alive
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures.append((r["id"], f"{type(e).__name__}: {e}"))
        print(f"  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*78}\nSWEEP COMPLETE")
    print(f"  ok       {len(results)}")
    print(f"  failed   {len(failures)}" + (f"  {failures}" if failures else ""))
    print(f"  skipped  {len(skipped)}" + (f"  {skipped}" if skipped else ""))
    print("\nNext: python -m poc.runner.compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
