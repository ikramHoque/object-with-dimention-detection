# Implementation Status

> **FILE PURPOSE** — How much of the POC actually exists, and how much of that has been
> proven to run. Separates *written* from *verified*, because those are different claims.
> Update on every change to pipeline behaviour.
>
> Map: `PIPELINE.md` · Open decisions: `GAPS.md`

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
| **Model adapters** verified | **3 of 13** — `grounding_dino`, `moge2`, `sam2` executed on real photographs, MPS |
| Defects found and fixed by testing | **7**, all real (see below) |
| Blocked on environment or data | end-to-end run |

**One-line read:** the arithmetic is trustworthy, the environment now works, and the two
default models execute on this Mac — but nothing has yet touched a real photograph, so no
accuracy claim exists.

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

## POC pipeline — `poc/pipelines/<name>/`

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
| `runner/compare.py` | Rank runs, pair anchor on/off | ✅ | run against 6 synthetic results |
| `runner/run_combination.py` | The orchestrator: one run → JSON, annotated frame, CSV | ✅ | executed end to end on real photographs |
| `models/detect_grounding_dino.py` | Baseline detector, Apache-2.0 | ✅ | executed on real photographs, MPS |
| `models/detect_owlv2.py` | Long-tail detector, Apache-2.0 | 🟡 | never executed |
| `models/detect_sam3.py` | Native masks, gated weights | 🔴 | never executed; new API, expect churn |
| `models/detect_rtdetr.py` | Real-time, licence-clean, fixed vocab | 🟡 | never executed |
| `models/detect_yolo_world.py` | ~20× faster — **AGPL, cannot ship** | 🟡 | never executed |
| `models/detect_yoloe.py` | Real-time + masks — **AGPL, cannot ship** | 🔴 | never executed; checkpoint names move |
| `models/depth_moge2.py` | Default depth, MIT | ✅ | executed on real photographs, MPS |
| `models/depth_anything3.py` | Multi-view depth, Apache variant | 🔴 | never executed; HF integration is new |
| `models/depth_unidepth.py` | Ceiling measurement — **CC BY-NC** | 🟡 | never executed |
| `models/segment_sam2.py` | Box → mask upgrade | ✅ | executed; door error 16.5% → 6.8%. Loads a `sam2_video` checkpoint — confirm before quoting |
| `models/classify_claude.py` | Method A, three model sizes | 🟡 | never executed — needs an API key |
| `runner/report.py` | Annotated frames, item table, CSV | ✅ | shared by CLI and notebook; output inspected |
| `runner/run_dataset_eval.py` | Detection scoring vs HomeObjects-3K | ✅ | 100 images: precision 0.695, recall 0.222 |
| `datasets/homeobjects.py` | Dataset loader, OpenCV not ultralytics | ✅ | 2,689 images loaded |
| `pipelines/_runner.py` | config.py → orchestrator | ✅ | both pipelines run |
| `pipelines/scaffold.py` | Create a pipeline folder | ✅ | key validation and licence warning tested |
| `pipelines/make_notebook.py` | Generate a pipeline's notebook | ✅ | both generated; one executed 14/14 cells |

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
| `poc/requirements.txt` | Pinned dependency set | ✅ | installed; MoGe pinned out of the resolve |
| `poc/setup.sh` | Creates `.venv` on Python 3.12 via uv, registers kernel, verifies imports | ✅ | run end to end, exit 0 |
| `poc/README.md` | Folder map and the run command | ✅ | rewritten for the pipelines layout |
| `PIPELINE.md` | The nine stages with the function behind each, Method A vs B, worked example | ✅ | **start here** |
| `RUNNING.md` | Every way to run it, timings, troubleshooting | ✅ | replaces RUNBOOK + RUN-LOCAL + COLAB |
| `LEARN.md` | NMS, IoU, thresholds — what they do and how to tune | ✅ | lessons 2-8 still to write |
| `poc/pipelines/README.md` | What belongs in a pipeline folder, and what must not | ✅ | |
| `MODELS.md` | All 13 models, licences, and how each one thinks | ✅ | absorbed the former ARCHITECTURES.md |
| `poc/pipelines/*/config.py` | 14 one-factor-at-a-time combinations, each with the question it answers | ✅ | replaced combinations.json; the full grid would be 288 runs / ~19h |
| `poc/results/` · `poc/pipelines/*/results/` | Per-run JSON, annotated frames, CSV | ✅ | populated; gitignored |
| `GAPS.md` | 22 open gaps in three decision groups | ✅ | A4 and A5 updated with measured sensitivity |
| `poc/pipelines/*/data/input/` | Per-pipeline drop zone for footage | ✅ | gitignored. Still **no customer footage** — only a dataset image |


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
| ~~1~~ | ~~No Python environment~~ | **DONE 3 Sept 2026.** `poc/.venv` on Python 3.12, torch 2.14.0 on **mps**, all 11 core imports verified by `setup.sh` itself | — |
| 1b | Optional models not installed | `python -m poc.models.registry` shows what's missing. SAM 3 needs `huggingface-cli login`; Ultralytics is AGPL and deliberately not installed | varies |
| 2 | No room footage | Shoot one bedroom and one living room, slow pan, door visible in frame | 30 min |
| 3 | No ground truth | Copy the template, laser-measure the same two rooms | half a day |
| 4 | No API key | `export ANTHROPIC_API_KEY=...` — Method A only; Method B runs without it | 1 min |

Blocker 1 is now cleared. Blockers 2 and 4 get the pipeline running. **Blocker 3 is the one that makes the output
mean anything** — it is gap **A1**, and without it stage 9 prints nothing.

---

## Defects found by running the models (3 Sept 2026)

Found by executing Grounding DINO and MoGe-2 on this Mac for the first time. None of
these were visible from reading the code.

**3 · Open-vocabulary labels are token spans, not class names**

Grounding DINO returns the matched *text span*, whose shape depends on
`text_threshold`:

| `text_threshold` | returns | why |
|---|---|---|
| 0.35 / 0.25 | `''` | no token cleared the bar |
| 0.05 | `'sideboard rolled rug mattress cardboard box door'` | many did |
| in the band | `'sofa'` | what we want |

Neither off-nominal form is a key in `detect_vocab.json`, so `allowed` became `None`
and `nearest_class()` searched **all 42 size classes** instead of 2–3 — worth about
18 points of accuracy by its own docstring, lost with no error and nothing in the
output JSON. Fixed by `normalise_label()` in `poc/models/base.py`; counted per-reason
in the results under `label_quality`.

**4 · The scale anchor discarded real furniture**

`if "door" in d.label` is a substring test. Against the span
`'sofa wardrobe door chair'` it matched, so a sofa was skipped as though it were the
door anchor. Now an exact `== "door"` comparison, which is safe because `door` is the
only door-like prompt in the 26.

**5 · Unnamed boxes double-counted objects** *(introduced while fixing 3, caught by testing)*

`nms()` groups by exact label, so an unnamed box sitting on a named box was treated as
a second object and both survived — inflating volume, which is the precise failure this
project exists to fix. `nms()` now runs a second cross-label pass dropping unnamed boxes
that shadow named ones, while still preserving genuine overlaps (a chair in front of a
sofa is two objects, and there is a regression test for exactly that).

**6 · `nms()` output order was nondeterministic**

`for label in {d.label for d in dets}` iterates a set of **strings**, and Python
randomises string hashing per process. Three runs on identical input produced three
different orders. Volume totals were unaffected (order-independent sums), but every
`measurements` block in every results JSON came out shuffled — so two runs could not
be diffed, and any comparison pairing rows by position was silently wrong. It caught
me out directly: my first masked-vs-unmasked comparison paired the wrong objects.
Fixed by iterating `sorted()` and giving the output a total order (score desc, then
box, then label).

For a project whose entire method is comparing 14 combinations, reproducible ordering
is not cosmetic.

---

## First real measurement: what SAM 2 masks actually do (7 Sept 2026)

Same photograph, same detector and depth model, `--segmenter sam2` the only change.
Objects matched by detection score, not row position.

| object | no mask (w×d×h) | with mask | m³ no-mask | m³ mask |
|---|---|---|---|---|
| (unnamed) sofa | 2.68 × 1.74 × 0.75 | 2.00 × 0.69 × 0.70 | 3.486 | 0.962 |
| (unnamed) | 2.22 × 0.76 × 0.64 | 1.00 × 0.94 × 0.49 | 1.086 | 0.458 |
| armchair | 0.90 × 0.31 × 0.73 | 0.47 × 0.22 × 0.70 | 0.205 | 0.074 |
| tv stand | 1.93 × 1.07 × 0.74 | 1.27 × 0.36 × 0.60 | 1.522 | 0.272 |
| rolled rug | 2.01 × 0.99 × 0.53 | 1.57 × 0.92 × 0.03 | 1.057 | 0.048 |
| chest of drawers | 1.93 × 1.02 × 0.73 | 1.27 × 0.36 × 0.60 | 1.438 | 0.272 |

**The clear win is the scale anchor.** The door is measured from its own pixels instead
of a rectangle containing wall:

```
no mask   door 2.307 m vs 1.981 m true   -> +16.5% error
with mask door 2.116 m vs 1.981 m true   ->  +6.8% error
```

Cubed, that is the difference between a 58% and a 22% volume inflation, applied to the
whole room. On this evidence masks should be **on by default** for the anchor alone.

**But masks trade one error for another.** Gross over-measurement becomes systematic
under-measurement: 0.36 m deep for a chest of drawers (real: 0.45–0.50 m) and 0.22 m
for an armchair. The camera only sees front faces, so a mask measures the front face
faithfully — which is the single-view depth limit, gap **B8**, arriving in real data.

**And our guard did not fire.** `depth_observability` reported `class_prior 0/6 (0%)`
despite those figures. `extent()` flags degeneracy at `pca_ratio < 0.12 or d < 0.05`,
so it catches only total collapse (under 5 cm), not a 30–40% systematic shortfall.
**The threshold is too lenient and needs recalibrating against measured rooms** — which
needs gap A1 first, because there is currently nothing to calibrate against.

**A domain finding worth keeping:** the rug measured 1.57 × 0.92 × **0.03** m — correct
for a rug lying flat, and irrelevant for a removal quote, because it ships **rolled**.
Measured dimensions are not packed dimensions. The cube table already encodes the
packed form (`rug_rolled`), so for some classes the measurement should be discarded in
favour of the table rather than reconciled with it. That distinction is not yet in the
pipeline.

**Caveat on this run:** transformers warned it loaded a `sam2_video` checkpoint into
`Sam2Model`. The masks are plainly working, but the checkpoint choice in
`poc/models/segment_sam2.py` should be confirmed before these numbers are quoted.

---

**7 · Ambiguous labels threw away information — found by LOOKING at the output**

Once `run_combination` started drawing boxes, the picture showed `rug_rolled` and
`sofa_3_seat` both sitting on the **coffee table**. Tracing it back: the table was
detected with raw text `'coffee table side table'`, both table prompts, so the
"refuse to guess when ambiguous" rule left it unnamed — and `nearest_class()` then
searched all 42 classes by geometry and chose **`sofa_3_seat`, 1.416 m³**, for an
object of roughly 0.3 m³.

The earlier reasoning — *a wrong label is worse than none* — is right for `sofa`
versus `cardboard box`. It is wrong when the tied prompts **agree on size**, because
then the ambiguity has no bearing on the quote and collapsing it discards information
we already had.

Fixed by `P.allowed_classes()`, which searches the **union** of the candidates'
classes instead of nothing:

```
before  unnamed -> all 42 -> sofa_3_seat                            1.416 m³
after   {bedside_table, coffee_table, side_table} -> coffee_table   0.340 m³
```

**Measured effect on the room total: 5.748 → 4.672 m³, a 19% reduction.** Same image,
same models, same thresholds; the only change is how tightly the search is restricted.

Two lessons worth keeping:

1. **The annotated image is a debugging tool, not decoration.** This error was
   invisible in the JSON, where every row looked plausible. It was obvious the moment
   the boxes were drawn on the photograph.
2. Restricting the search is worth ~18 points, but restricting it **wrongly** is worse
   than not restricting it. The union is the honest middle: narrower than the whole
   table, and it invents nothing.

---

## Human-readable output (7 Sept 2026)

`poc/runner/report.py` — shared by the CLI and the notebook, so both draw identically.
Each run now writes `poc/results/<run>/`:

- `frame_NNN.jpg` — a box per object labelled with class, W×D×H and m³. Colour encodes
  provenance: green measured, amber depth assumed from the cube table, red unnamed,
  blue the door reference. A picture full of amber and red means the volume rests on
  assumptions, which is legible at a glance rather than buried in JSON.
- `items.csv` — per-object measurements plus the inventory. CSV because a surveyor
  reviews this in Excel.
- the same item table printed to the console.

`.gitignore` widened to `poc/results/**`: those JPEGs embed the input photograph, so a
dataset input would otherwise commit dataset imagery.

---

### A design decision worth recording

When a span fuses several prompts we return **no label** rather than the longest match.
A wrong label is worse than none here: with no label, `nearest_class()` searches the
full table using the **measured** w/d/h and degrades gracefully; with a wrong label it
is forced into that class's 2–3 candidates whatever the tape says. The first version of
the fix returned `max(hits, key=len)`, which turned a real span into `'cardboard box'`
on string length alone — confident nonsense, the one output this pipeline must never
produce.

---

## Per-pipeline folders (7 Sept 2026)

`poc/pipelines/<name>/` now holds everything belonging to one pipeline — `config.py`,
`run.py`, `notebook.ipynb`, `data/input/`, `results/`, `README.md` — while the stage
logic stays shared in `poc/runner/` and `poc/models/`.

**The design decision:** config, entry points and data are per pipeline; **logic is
not copied.** Copying it would make each folder self-contained and wreck the
codebase, because fourteen copies drift and then nobody can tell which is right.
Two pieces of local evidence rather than a principle: `poc/pipeline.ipynb` carried an
inline copy of the stage logic, was never run, and had drifted into disagreeing with the
CLI — it has since been deleted for that reason; and the
coffee-table 19% over-estimate survived partly because the notebook kept its own
vocabulary lookup.

Notebooks are **generated** by `make_notebook.py` for the same reason — the narration
is identical across pipelines and differs only in the model names, so one source
regenerates all of them. That generator previously existed only in a scratch
directory, meaning nobody but its author could regenerate the notebook; it is now in
the repo.

Enabling changes:

- `run_combination` takes `--input-dir` / `--results-dir`, defaulting to the shared
  folders, so a pipeline can own its input and output.
- `run_combination` takes `--box-th` / `--txt-th`, which **previously had nowhere to
  go**: the detector was built with no arguments, so every run silently used the
  adapter default and a config could not set a threshold at all.
- `registry.build()` drops keyword arguments a model cannot accept, with a warning.
  RT-DETR has no text threshold because it has no text; raising would force every
  caller to special-case each model, and ignoring silently would let a threshold you
  thought you set do nothing.

Verified: the per-pipeline CLI produces **4.672 m³** on the demo room, identical to
the general CLI, with JSON, annotated frame and `items.csv` all inside the pipeline's
own `results/`; the shared `poc/results/` was untouched. Multi-input refuses to guess
and lists the options. The scaffolder rejects unknown model keys and warns when a
combination cannot ship.

**Known limitation:** one run is one room. Several rooms need several invocations, and
each re-pays the detector's ~370 s warm-up. Batching is not implemented.

A `.gitignore` subtlety worth recording: `poc/pipelines/*/data/**` matched the
`data/input` **directory** and excluded it, and git will not descend into an excluded
directory to re-include a file — so `data/input/.gitkeep` vanished and a fresh clone
had no input folder. Ignoring the *contents* (`data/input/*`) keeps the folder.

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
