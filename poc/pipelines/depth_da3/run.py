"""
FILE PURPOSE
    CLI entry point for the depth_da3 pipeline.

HOW TO RUN
    python -m poc.pipelines.depth_da3.run --input my_room.jpg
    python -m poc.pipelines.depth_da3.run                     # if data/input has one thing
    python -m poc.pipelines.depth_da3.run --input my_room.jpg --box-th 0.20

    Input comes from   poc/pipelines/depth_da3/data/input/
    Output goes to     poc/pipelines/depth_da3/results/

    It is thin on purpose: config.py says what to run, poc/pipelines/_runner.py
    handles the plumbing, and poc/runner/run_combination.py does the work — the
    same code path as the general-purpose CLI, so this cannot drift from it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.pipelines._runner import run_from_config      # noqa: E402
from poc.pipelines.depth_da3 import config                # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_from_config(config))
