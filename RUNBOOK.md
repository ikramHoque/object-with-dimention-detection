# Runbook — every command, and which file it runs

> **FILE PURPOSE** — The single place to look up how to run anything in this project.
> Every runnable file also carries a `HOW TO RUN` block in its own docstring.
>
> Local setup and troubleshooting: `RUN-LOCAL.md` · GPU options: `COLAB.md`

---

## Before any command

```bash
cd /Users/ikramul/Workspace/Office-Workspace/RND/Nippon
source poc/.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1        # required on Apple Silicon
```

All commands are run from the **repo root**, not from inside `poc/`.

---

## Quick reference

| Command | File that runs | What you get |
|---------|----------------|--------------|
| `./poc/setup.sh` | `poc/setup.sh` | Python 3.12 venv + all dependencies |

> **New to this?** Read `PIPELINE.md` first. It walks the nine stages with the
> function and file behind each one, and explains why `run_dataset_eval` (detection
> metrics) and `run_combination` (the actual pipeline) are different tools.

| `python -m poc.models.registry` | `poc/models/registry.py` | all 13 models, licences, install status |
| `python -m poc.datasets.homeobjects --download` | `poc/datasets/homeobjects.py` | the labelled dataset (~390 MB) |
| `python -m poc.runner.run_dataset_eval` | `poc/runner/run_dataset_eval.py` | **detection accuracy vs ground truth** |
| `python -m poc.runner.run_combination` | `poc/runner/run_combination.py` | one combination on one room → JSON |
| `python -m poc.runner.run_sweep` | `poc/runner/run_sweep.py` | all 14 combinations |
| `python -m poc.runner.compare` | `poc/runner/compare.py` | ranked results table |
| `python -m poc.rnd.run_research_ceiling` | `poc/rnd/run_research_ceiling.py` | non-shippable models, quarantined |
| `jupyter lab poc/pipeline.ipynb` | `poc/pipeline.ipynb` | the visual pipeline, stage by stage |
| *(upload to Colab)* | `poc/colab/baseline_colab.ipynb` | the baseline on a free T4 |

### Files that are never run directly

| File | How it's used instead |
|------|----------------------|
| `poc/models/base.py` | the contracts. Imported by everything |
| `poc/models/detect_*.py` | selected by `--detector <key>` |
| `poc/models/depth_*.py` | selected by `--depth <key>` |
| `poc/models/segment_sam2.py` | selected by `--segmenter sam2` |
| `poc/models/classify_claude.py` | selected by `--classifier <key>` |
| `poc/runner/pipeline.py` | the stage maths. Imported by the runners |
| `poc/cube_table.json` | data: size class → volume |
| `poc/detect_vocab.json` | data: what the detectors look for |
| `poc/combinations.json` | data: the 14 sweep presets |

---

## 1 · Setup — once

```bash
./poc/setup.sh
source poc/.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
**Runs:** `poc/setup.sh` · ~20 min, ~2.5 GB. Creates `poc/.venv` on Python 3.12 and
registers the **NX Survey POC** Jupyter kernel.

```bash
python -m poc.models.registry
```
**Runs:** `poc/models/registry.py` · Confirms what's installed. Everything except the AGPL
and non-commercial models should say `yes`.

---

## 2 · Detection accuracy — no footage needed

This is the fastest route to a real number, because the dataset comes with labels.

```bash
python -m poc.datasets.homeobjects --download
```
**Runs:** `poc/datasets/homeobjects.py` · ~390 MB, once. Note `images_with_a_door` — that's
how many images can exercise the scale anchor.

```bash
python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50
```
**Runs:** `poc/runner/run_dataset_eval.py` · 2–4 min on an M1.
Read the output in this order: **`door` recall → `wardrobe` recall → overall recall.**

```bash
python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30
```
Sweeps `box_th` across six values. Watch precision fall as recall rises. **Pick the row you
want, not the best F1** — for a survey, recall wins.

```bash
python -m poc.runner.run_dataset_eval --detector grounding_dino --depth moge2 --limit 20
```
Adds a depth sanity check. **Diagnostic only** — this dataset has no depth labels.

```bash
for d in grounding_dino owlv2 rtdetr; do
  python -m poc.runner.run_dataset_eval --detector $d --limit 50 --save
done
```
Same images, three shippable detectors. `rtdetr` will score lower — it's closed-vocabulary
and has no class for 9 of our 26 prompts. That's the point of including it.

---

## 3 · A real room — the actual pipeline

Needs your own footage **and** hand measurements.

```bash
# footage: slow pan, door visible, 30-60s per room
cp ~/Desktop/bedroom.mp4 poc/data/input/

# ground truth: the step that makes this an experiment (gap A1)
cp poc/ground_truth_template.csv poc/ground_truth.csv
# ... fill it in with a laser measure

export ANTHROPIC_API_KEY=sk-ant-...          # Method A only
```

```bash
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5
```
**Runs:** `poc/runner/run_combination.py` · 1–2 min. Writes one JSON to `poc/results/`.

```bash
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 --no-anchor
```
**The key experiment.** Identical, scale anchor off. The gap between these two runs is the
entire value of the anchor.

### Useful variations

| Add this | Effect |
|----------|--------|
| `--segmenter sam2` | upgrade boxes to masks, removing background pixels |
| `--classifier none` | geometry only, no language model, no API cost |
| `--classifier claude_opus5` | stronger classifier, ~2.5× cost |
| `--dedup median` | reject false positives instead of surviving occlusion |
| `--max-frames 8` | half the work, for a quick check |
| `--help` | every option |

---

## 4 · All 14 combinations

```bash
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --dry-run
```
**Runs:** `poc/runner/run_sweep.py` · Prints the plan with licence and install warnings.
Executes nothing. **Always do this first.**

```bash
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --skip-unavailable
```
Executes it. 1–2 hours on an M1. A failing run doesn't stop the sweep.

```bash
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --only anchor_off,det_sam3
```
Just those two. Ids come from `poc/combinations.json`.

---

## 5 · Read the results

```bash
python -m poc.runner.compare --room BED01
```
**Runs:** `poc/runner/compare.py` · Ranks every run by bias, pairs anchor on/off runs with
identical config, and flags anything that cannot ship.

```bash
python -m poc.runner.compare --method b --sort abs
```
Rank Method B instead, by absolute error.

**Three numbers matter:**

| Field | Read it how |
|-------|-------------|
| `vol_bias_pct` | **first.** Consistent bias is one multiplier from fixed |
| `vol_abs_err_pct` | second. Random spread is the expensive problem |
| `class_prior_pct` | high = Method B took depth from the cube table, not the camera (gap B8) |

---

## 6 · The models that cannot ship

```bash
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 --dry-run
```
**Runs:** `poc/rnd/run_research_ceiling.py` · Shows the plan and licence warnings.

```bash
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 \
    --i-have-legal-clearance
```
Runs YOLO-World and UniDepthV2 to measure what staying licence-clean costs. Output goes to
`poc/rnd/results/`, tagged `shippable: false`. **Get NX legal's view first** — Ultralytics'
position covers internal R&D. See `poc/rnd/README.md`.

---

## 7 · Notebooks

```bash
jupyter lab poc/pipeline.ipynb
```
**Runs:** `poc/pipeline.ipynb` · Select the **NX Survey POC** kernel. Shows keyframes,
detection boxes and depth maps at each stage — much better than the CLI for judging *why*
something went wrong.

**Colab:** upload `poc/colab/baseline_colab.ipynb`, set **Runtime → T4 GPU**, run top to
bottom. Naming convention is `poc/colab/<combination_id>_colab.ipynb`.

---

## Which file do I edit to change…

| I want to change | Edit |
|------------------|------|
| what the detectors look for | `poc/detect_vocab.json` |
| a size class's volume | `poc/cube_table.json` |
| which combinations the sweep runs | `poc/combinations.json` |
| detection thresholds (per run) | `--box-th` / `--txt-th`, or `CFG` in the notebook |
| the NMS overlap threshold | `nms(dets, iou_thresh=...)` in `poc/models/base.py` |
| frame sampling and blur rejection | `CFG` in the notebook, or `load_keyframes()` args |
| how objects are measured | `extent()` in `poc/runner/pipeline.py` |
| the depth fallback for flat surfaces | `resolve_dims()` in `poc/runner/pipeline.py` |
| the scale anchor logic | `door_scale()` in `poc/runner/pipeline.py` |
| the Method A prompt | `PROMPT` in `poc/models/classify_claude.py` |
| add a whole new model | new `poc/models/<kind>_<name>.py` + one line in `registry.py` |

---

## Environment variables

| Variable | When | Why |
|----------|------|-----|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | **always, on Mac** | some ops have no Metal kernel; without this they crash |
| `NX_DEVICE=cpu` | when MPS still fails | forces CPU for every model. Accepts `cpu`/`mps`/`cuda` |
| `ANTHROPIC_API_KEY` | Method A only | Method B runs without it |
| `HF_TOKEN` *(or `huggingface-cli login`)* | for SAM 3 | its weights are gated |

---

## Typical first session, start to finish

```bash
# setup
./poc/setup.sh && source poc/.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m poc.models.registry

# a real number in under 10 minutes, no footage required
python -m poc.datasets.homeobjects --download
python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50

# see how the parameters behave
python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30

# then, with your own footage + measurements
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 --no-anchor
python -m poc.runner.compare --room BED01
```
