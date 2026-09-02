# Implementation Status

**Snapshot — 2 Sept 2026**

The POC pipeline is **written end to end and has never been run end to end.** All nine
stages plus an appendix exist as working code, the notebook is structurally valid, and
every code cell parses. What is missing is not code — it is a Python environment, some
room footage, and hand measurements to score against.

| | |
|---|---|
| Pipeline stages written | **10 / 10** (9 + appendix) |
| Code | 472 lines, 15 cells, 16 functions |
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
| **5** Arm B | PCA-oriented 3D extent per object → nearest size class | 46 loc | 🟢 | `nearest_class` replica: 5/5 correct | stage 3 for `extent` |
| **6** Arm A | Claude vision → forced size-class JSON | 63 loc | 🟡 | syntax only | env, API key, footage |
| **7** Dedup | Collapse the same object across frames | 26 loc | 🟢 | replica: occlusion + false-positive cases pass | none (needs 6) |
| **8** Aggregate | Three volume figures side by side | 17 loc | 🟢 | replica: living 3.99 m³, bedroom 4.84 m³ | none (needs 5, 7) |
| **9** Evaluate | Score vs ground truth; bias split from spread | 52 loc | 🟡 | syntax only | env, `ground_truth.csv` |
| **A** Appendix | Scale-error sensitivity simulation | 46 loc | ✅ | **executed from notebook source** | none |

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
| 2 | No room footage | Shoot one bedroom and one living room, slow pan, door visible in frame | 30 min |
| 3 | No ground truth | Copy the template, laser-measure the same two rooms | half a day |
| 4 | No API key | `export ANTHROPIC_API_KEY=...` — Arm A only; Arm B runs without it | 1 min |

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
3. A verdict on **Arm A vs Arm B**, so one can be dropped.
4. A list of items the detector vocabulary misses entirely.

None of those need more code than already exists. They need the four blockers cleared.

---

## Keeping this file honest

Update on every merge that changes pipeline behaviour. Two rules:

- A stage moves to ✅ **only when the cell itself has run on real footage** — not when a
  replica passes and not when the syntax checks out.
- If a status is uncertain, mark the worse one.
