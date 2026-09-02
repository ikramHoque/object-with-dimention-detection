# POC — can a model identify room contents and how much space they take?

One notebook, two competing methods, scored against hand measurements.

## The question

Given a photo or video of a room, can we produce a reliable **total volume** for what
is in it? Two candidate methods, run side by side:

| | Method | How volume is obtained |
|---|---|---|
| **Arm A** | Classify → look up | Decide "3-seater sofa", read packed volume off `cube_table.json` |
| **Arm B** | Measure geometrically | Metric depth → 3D extent of each object → compute volume |

Both incumbents in this market use Arm A and neither measures objects. This POC checks
that on our own data instead of taking it on trust.

## Setup

```bash
./setup.sh
export ANTHROPIC_API_KEY=sk-ant-...      # Arm A only
```

Drop input files into `data/input/`. Images (`.jpg .png`) or video (`.mp4 .mov`).

```bash
source .venv/bin/activate && jupyter lab pipeline.ipynb
```

## Before the numbers mean anything

Fill in `ground_truth.csv` (copy `ground_truth_template.csv`) for the rooms you shoot.
Without it the notebook still runs and still prints volumes — but nothing tells you
whether they are right. See gap **A1** in `../GAPS.md`. Half a day with a laser measure
is what turns this from a demo into an experiment.

## Notebook stages

| Stage | Does | Notes |
|-------|------|-------|
| 1 Ingest | Video/image → sharp keyframes | Blur + brightness gate |
| 2 Detect | Open-vocab boxes per frame | Grounding DINO |
| 3 Depth | Metric point map in metres | MoGe-2, MIT licence |
| 4 Scale anchor | Correct scale from a known reference | **The ablation switch — gap A5** |
| 5 Arm B | 3D extent per object → volume | Robust percentile extent |
| 6 Arm A | Size class per object → table lookup | Claude vision, forced JSON |
| 7 Dedup | Collapse the same object across frames | Max-per-frame heuristic, gap B1 |
| 8 Aggregate | Room totals, both arms | |
| 9 Evaluate | Score against ground truth | Class acc, count err, dim MAPE, volume MAPE + bias |

## What to look at first

Stage 9 prints **bias** separately from **spread**. Bias is the one that matters —
a system that is consistently 15% low is fixable with a coefficient; one that is
randomly ±15% is not.

Then run Stage 4 twice, anchor on and off, and compare. That delta is the most
valuable number the POC produces.
