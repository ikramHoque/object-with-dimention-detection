# Implementation Status

> **FILE PURPOSE** — How much of the POC actually exists, and how much of that has been
> proven to run. Separates *written* from *verified*, because those are different claims.
> Update on every change to pipeline behaviour.
>
> Map: `ARCHITECTURE.md` · Open decisions: `GAPS.md`

**Snapshot — 3 Sept 2026**

Two things exist now: the exploratory **notebook** (one fixed pipeline, nine stages) and a
**pluggable model harness** (13 interchangeable models, 14 curated combinations, a licence
gate and a ranking tool).

The picture has changed since the last snapshot. The pipeline **maths is now genuinely
tested** — executed against synthetic geometry with real numpy, all assertions passing.
What remains unverified is the **model integrations**, which need weights and footage.

| | |
|---|---|
| Notebook | 28 cells, 864 lines, 9 stages + appendix |
| Harness | 19 modules, 2196 lines |
| Models available | **13** — 6 detectors, 3 depth, 1 segmenter, 3 classifiers |
| Models that cannot ship | **3** — `yolo_world`, `yoloe` (AGPL), `unidepth2` (CC BY-NC) |
| Curated combinations | 14, one-factor-at-a-time |
| **Algorithm** verified by execution | **9 of 9 functions** — nms, extent, door_scale, nearest_class, resolve_dims, dedup ×2, score, volume |
| **Tooling** verified by execution | registry, licence gate, sweep planner, ranking |
| **Model adapters** verified | **0 of 13** — no weights, no deps installed |
| Defects found and fixed by testing | **2**, both real (see below) |
| Blocked on environment or data | end-to-end run |

**One-line read:** the arithmetic is trustworthy and the harness works; nothing has yet
touched a real photograph.

---|---|
| Pipeline stages written | **10 / 10** (9 + appendix) |
| Code | 524 lines of logic + 340 lines of explanatory comment, 15 cells, 16 functions |
| Executed as written | **1 / 10 stages** (the appendix) |
| Logic verified via standalone replica | 3 stages (partial) |
| Statically validated | 15 / 15 cells — JSON, syntax, name resolution |
| Blocked on environment or data | 7 / 10 stages |
| Product scope beyond the POC | not started, by design — see `GAPS.md` Group C |

**One-line read:** implementation is roughly complete for POC scope; *verification* is at
about 15%, and closing that gap is four setup steps, not more engineering.

---

## Legend

| Mark | Meaning |
|------|---------|
| ✅ | Written **and executed successfully as written** |
| 🟢 | Written; logic proven by a standalone replica, but the cell itself never run |
| 🟡 | Written and syntax-valid; never executed |
| 🔴 | Written but expected to need fixing on first real run |
| ⬜ | Not started |

---

## POC pipeline — `poc/pipeline.ipynb`

| Stage | Does | Code | Status | Verified by | Blocked on |
|-------|------|------|--------|-------------|------------|
| **0** Config | Paths, model IDs, thresholds, ablation switches | 45 loc | 🟡 | syntax only | env |
| **1** Ingest | Video/image → sharp keyframes; blur + brightness gate | 55 loc | 🟡 | syntax only | env, footage |
| **2** Detection | Grounding DINO open-vocab boxes, 26 prompts | 52 loc | 🔴 | syntax only | env, weights, footage |
| **3** Depth | MoGe-2 metric point map in metres | 38 loc | 🔴 | syntax only | env, weights, footage |
| **4** Scale anchor | Door-height correction factor — **the ablation switch** | 32 loc | 🔴 | syntax only | stages 2 + 3 |
| **5** Method B | PCA-oriented 3D extent per object → nearest size class | 46 loc | 🟢 | `nearest_class` replica: 5/5 correct | stage 3 for `extent` |
| **6** Method A | Claude vision → forced size-class JSON | 63 loc | 🟡 | syntax only | env, API key, footage |
| **7** Dedup | Collapse the same object across frames | 26 loc | 🟢 | replica: occlusion + false-positive cases pass | none (needs 6) |
| **8** Aggregate | Three volume figures side by side | 17 loc | 🟢 | replica: living 3.99 m³, bedroom 4.84 m³ | none (needs 5, 7) |
| **9** Evaluate | Score vs ground truth; bias split from spread | 52 loc | 🟡 | syntax only | env, `ground_truth.csv` |
| **A** Appendix | Scale-error sensitivity simulation | 46 loc | ✅ | **executed from notebook source** | none |

## Model harness — `poc/models/` and `poc/runner/`

| Component | Does | Status | Verified by |
|-----------|------|--------|-------------|
| `models/base.py` | The contracts that make models swappable | ✅ | `Detection`, `nms` executed |
| `models/registry.py` | 13 models, availability probing, **licence gate** | ✅ | CLI run; gate confirmed to refuse `yolo_world` |
| `runner/pipeline.py` | Model-agnostic stage maths | ✅ | full assertion suite passes on synthetic geometry |
| `runner/run_sweep.py` | Plan and execute the 14 combinations | 🟢 | dry-run plan verified; execution needs models |
| `runner/compare.py` | Rank runs, pair anchor on/off | ✅ | run against 6 synthetic results |
| `runner/run_combination.py` | One combination → one result JSON | 🟡 | CLI + licence gate verified; body needs models |
| `models/detect_grounding_dino.py` | Baseline detector, Apache-2.0 | 🟡 | never executed |
| `models/detect_owlv2.py` | Long-tail detector, Apache-2.0 | 🟡 | never executed |
| `models/detect_sam3.py` | Native masks, gated weights | 🔴 | never executed; new API, expect churn |
| `models/detect_rtdetr.py` | Real-time, licence-clean, fixed vocab | 🟡 | never executed |
| `models/detect_yolo_world.py` | ~20× faster — **AGPL, cannot ship** | 🟡 | never executed |
| `models/detect_yoloe.py` | Real-time + masks — **AGPL, cannot ship** | 🔴 | never executed; checkpoint names move |
| `models/depth_moge2.py` | Default depth, MIT | 🟡 | never executed |
| `models/depth_anything3.py` | Multi-view depth, Apache variant | 🔴 | never executed; HF integration is new |
| `models/depth_unidepth.py` | Ceiling measurement — **CC BY-NC** | 🟡 | never executed |
| `models/segment_sam2.py` | Box → mask upgrade | 🟡 | never executed; has a filled-box fallback |
| `models/classify_claude.py` | Method A, three model sizes | 🟡 | never executed |

## Two defects the tests caught

Both real, both found before any footage existed, both fixed and verified.

**1 · The door measurement was biased ~4% high.** `extent()` clips to 2nd–98th percentiles
to reject depth outliers. The door anchor inherited that, but a door box is tight and has
few outliers — so clipping was shaving ~4% off the true height, inflating the scale factor
by 4% and the volume by ~13%. A systematic bias, in the exact term the anchor exists to
remove. Fixed with wider percentiles for the door: **scale error 4.2% → 1.0%**, volume bias
~13% → 3.1%.

**2 · Cameras only see the front of things.** Not predicted. A camera never sees the back
of a wardrobe, so a flat-fronted object's point cloud is a thin sheet with no measurable
depth, and the computed volume collapsed toward zero. This is a hard limit of single-view
geometry affecting most furniture. `resolve_dims()` now detects it, matches on the two
observed dimensions, and adopts a class prior — verified to resolve a wardrobe-shaped
surface to `wardrobe_double` at 0.60 m rather than 0.00 m.

**The second one has a consequence to watch.** Every run records `class_prior_pct`. When
depth comes from the cube table, Method B is using Method A's data and the two are no
longer independent. If that fraction is high on real footage, pure geometric measurement
is not viable from single-view capture — itself a headline finding. Tracked as gap **B8**.

### Why three stages are marked 🔴

Not because they are wrong — because they touch fast-moving external APIs and unverified
assumptions that will only surface on a real frame:

- **Stage 2** — `post_process_grounded_object_detection` renamed `box_threshold` → `threshold`
  around transformers 4.51. A fallback is already in place, but the label key also moved
  (`labels` vs `text_labels`) and is handled defensively. Expect one round of fixing.
- **Stage 3** — MoGe-2's point-map axis convention is assumed to put the vertical on axis 1.
  The cell prints the per-axis spans and a warning so this can be checked on frame one.
  If the Y span reads like room width, flip `VERT_AXIS`.
- **Stage 4** — depends entirely on stage 3's axis assumption being right, and on a door
  actually being visible and detected in the footage.

---

## Supporting artefacts

| File | Purpose | Status | Note |
|------|---------|--------|------|
| `poc/cube_table.json` | 42 size classes → packed volume + typical dimensions | ✅ | **PLACEHOLDER** — must be replaced with NX's own cube sheet (gap A3) |
| `poc/detect_vocab.json` | 26 detector prompts → candidate size classes | ✅ | `door` included as scale anchor, excluded from inventory |
| `poc/ground_truth_template.csv` | Schema + 10 example rows | ✅ | template only; the real file does not exist yet |
| `poc/requirements.txt` | Pinned dependency set | 🟡 | never installed |
| `poc/setup.sh` | Creates `.venv` on Python 3.12 via uv, registers kernel | 🟡 | never run |
| `poc/README.md` | Run order, stage table, what to look at first | ✅ | |
| `ARCHITECTURE.md` | Diagrams, stage I/O, model table, worked example | ✅ | start here — explains Method A vs Method B |
| `MODELS.md` | All 13 models, licences, the YOLO answer, how to sweep | ✅ | read before choosing a model |
| `poc/combinations.json` | 14 one-factor-at-a-time presets, each with its question | ✅ | full grid would be 288 runs / ~19h |
| `poc/results/` | One JSON per run | ⬜ | empty; gitignored |
| `GAPS.md` | 22 open gaps in three decision groups | ✅ | A4 and A5 updated with measured sensitivity |
| `poc/data/input/` | Drop zone for footage | ⬜ | **empty** |
| `poc/data/output/` | Artefact output | ⬜ | empty |

---

## What has actually been executed

Being precise, because "it's written" and "it works" are different claims.

**Executed from the notebook's own source:**
- Appendix sensitivity simulation — full cell, real output.

**Executed as a standalone replica** (same algorithm, numpy calls swapped for `math` /
`statistics` because numpy is not installed):
- `dedup_counts` — confirmed `max` keeps an occlusion peak and admits a 1-of-8 false
  positive; `median` does the reverse. Behaves as documented.
- `vol_from_inventory` — a typical living room totals 3.99 m³ / 141 ft³, a bedroom
  4.84 m³ / 171 ft³. Plausible against trade norms.
- `nearest_class` — 5/5 correct on hand-built cases, both detector-restricted and
  searching all 42 classes.

**Validated statically only:**
- Notebook JSON well-formed, nbformat 4.5, 28 cells.
- All 15 code cells parse.
- No undefined names across the cumulative namespace.

**Never executed in any form:**
- `load_keyframes`, `resize_long`, `sharpness`, `detect`, `det_load`, `draw`,
  `infer_depth`, `depth_load`, `door_scale`, `extent`, `classify_frame`, `pick_device`,
  and the stage 9 evaluation body.

### Result already produced

The appendix answers one design question without any input data. Class-assignment
accuracy against scale error:

| Scale error | Detector-narrowed | Whole cube table |
|---|---|---|
| 0% | 99.2% | 97.7% |
| 5% | 94.5% | 85.1% |
| **8%** ← MoGe-2 published | **90.2%** | **71.8%** |
| 15% | 77.5% | 50.8% |

Two conclusions, both folded back into `GAPS.md`: geometry must always be narrowed by the
detector label (worth ~18 points), and roughly one point of class accuracy is lost per
point of scale error — which is the measurable return on the stage 4 anchor work.

---

## Blockers to a first end-to-end run

In order. Nothing here is engineering.

| # | Blocker | Action | Effort |
|---|---------|--------|--------|
| 1 | No Python environment | `./poc/setup.sh` — pins 3.12, pulls ~2.5 GB of torch | 20 min |
| 1b | Optional models not installed | `python -m poc.models.registry` shows what's missing. SAM 3 needs `huggingface-cli login`; Ultralytics is AGPL and deliberately not installed | varies |
| 2 | No room footage | Shoot one bedroom and one living room, slow pan, door visible in frame | 30 min |
| 3 | No ground truth | Copy the template, laser-measure the same two rooms | half a day |
| 4 | No API key | `export ANTHROPIC_API_KEY=...` — Method A only; Method B runs without it | 1 min |

Blockers 1, 2 and 4 get the pipeline running. **Blocker 3 is the one that makes the output
mean anything** — it is gap **A1**, and without it stage 9 prints nothing.

---

## Not built, by design

Deferred POC scope, tracked as `GAPS.md` Group C. Listed so it is not mistaken for
oversight:

carton and box counts · contents of closed cupboards · coverage enforcement ·
dismantling and fragile-packing rules · browser upload UI · authentication ·
persistence and job history · back-office integration · GDPR/DPIA controls ·
photo-archive security · self-pack vs full-pack branching · multi-room aggregation ·
confidence intervals and P80 vehicle planning · staff review console

The browser upload UI is deliberately last. It carries no technical risk and is roughly a
day's work once the pipeline produces a trustworthy number.

---

## Definition of done for the POC

The POC is finished when it can state, with evidence:

1. A measured **bias** and **spread** for total room volume, on at least 6 rooms across
   3 properties.
2. The same figures with the scale anchor **on and off** — the delta that justifies or
   kills the anchor work.
3. A verdict on **Method A vs Method B**, so one can be dropped.
4. A list of items the detector vocabulary misses entirely.

None of those need more code than already exists. They need the four blockers cleared.

---

## Keeping this file honest

Update on every merge that changes pipeline behaviour. Two rules:

- A stage moves to ✅ **only when the cell itself has run on real footage** — not when a
  replica passes and not when the syntax checks out.
- If a status is uncertain, mark the worse one.
