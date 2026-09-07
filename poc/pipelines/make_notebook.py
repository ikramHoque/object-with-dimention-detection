"""
FILE PURPOSE
    Generates a pipeline's step-by-step notebook from its config.py.

HOW TO RUN
    python -m poc.pipelines.make_notebook grounding_dino__moge2
    python -m poc.pipelines.make_notebook --all

WHY GENERATED RATHER THAN HAND-WRITTEN
    Every pipeline needs the same nine stages with the same explanations, differing
    only in which models and thresholds they name. Hand-maintaining one notebook per
    pipeline guarantees they diverge: a fix to the stage-4 explanation would land in
    one and not the others, and notebooks are the hardest files to review a diff of.

    Generating them means the explanations have ONE source — this file — and
    regenerating propagates a fix to every pipeline at once.

    Note what is generated and what is not: the notebook's *narration and wiring*
    are generated; the *logic* it calls is imported from poc/runner/pipeline.py.
    Neither is copied per pipeline.

USED BY  you, after scaffold.py or after editing a config
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def build_cells(cfg) -> list[dict]:
    C: list[dict] = []

    def md(text):
        C.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})

    def code(text):
        C.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.rstrip("\n").splitlines(keepends=True)})

    name = cfg.NAME
    seg = getattr(cfg, "SEGMENTER", None)
    question = getattr(cfg, "QUESTION", name)

    md(f"""# {name} — one stage at a time

{question}

| | |
|---|---|
| detector | `{cfg.DETECTOR}` |
| depth | `{cfg.DEPTH}` |
| segmenter | `{seg or 'none'}` |

**This is the same pipeline `run.py` runs**, split into one cell per stage so you can
see what each produces before the next consumes it. It **imports** the stage
functions — it does not reimplement them — so the notebook and the CLI cannot
disagree. If a number here differs from the CLI, that is a bug.

**Why a notebook for tuning.** `{cfg.DETECTOR}`'s first inference in a process costs
about 370 s; every one after that costs ~2.7 s. The CLI pays that on every
invocation. A notebook keeps the kernel alive, so you pay it **once** and each
re-run afterwards is seconds — roughly 100x faster for threshold work.

Input comes from this folder's `data/input/`, results go to its `results/`.

| Stage | What it does | Function |
|---|---|---|
| 1 | frames in | `P.load_keyframes` |
| 2 | find objects | detector `.detect()` |
| 3 | metres per pixel | depth `.infer()` |
| 4 | **fix the scale** | `P.door_scale` |
| 5 | measure each object | `P.extent`, `P.resolve_dims` |
| 7-8 | dedup and total | `P.dedup_measured`, `P.vol_from_inventory` |
""")

    md("""## The only cell you edit

These start from `config.py`. Change them here to experiment; change `config.py` to
change the pipeline itself.
""")
    code(f"""import os
# Must be set before any torch op. Some operators have no Metal kernel and would
# hard-crash on MPS rather than falling back to the CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

INPUT      = None              # None = the only thing in data/input
BOX_TH     = {cfg.BOX_TH}               # keep a box at all.  LOWER -> more found, more junk
TEXT_TH    = {cfg.TEXT_TH}               # how sure of the NAME. see stage 2 for direction
USE_MASKS  = {bool(seg)}              # box -> mask. big win on stage 4
MAX_FRAMES = {getattr(cfg, 'MAX_FRAMES', 16)}                 # frames from a video or folder
DEDUP      = {getattr(cfg, 'DEDUP', 'max')!r}            # 'max' trusts each sighting; 'median' needs agreement""")

    md("""## Paths and imports

`HERE` is this pipeline's folder, found by looking for its `config.py`, so the
notebook works wherever Jupyter was started from.
""")
    code(f"""import sys, json
from pathlib import Path

PIPELINE = {name!r}
_cwd = Path.cwd().resolve()
HERE = next((q for q in (_cwd, *_cwd.parents) if (q / 'config.py').is_file()
             and q.name == PIPELINE), None)
if HERE is None:                      # started from elsewhere in the repo
    _root = next((q for q in (_cwd, *_cwd.parents)
                  if (q / 'poc' / 'runner' / 'pipeline.py').is_file()), None)
    assert _root is not None, f'cannot locate the repo from {{_cwd}}'
    HERE = _root / 'poc' / 'pipelines' / PIPELINE
ROOT = next(q for q in HERE.parents if (q / 'poc' / 'runner' / 'pipeline.py').is_file())
POC = ROOT / 'poc'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
from poc.runner import pipeline as P      # the maths — same module the CLI uses
from poc.runner import report as R        # drawing and tables — same as the CLI

VOCAB   = json.loads((POC / 'detect_vocab.json').read_text())['prompts']
PROMPTS = list(VOCAB.keys())
CLASSES = json.loads((POC / 'cube_table.json').read_text())['classes']
print(f'pipeline {{PIPELINE}}')
print(f'folder   {{HERE}}')
print(f'{{len(PROMPTS)}} prompts, {{len(CLASSES)}} size classes')""")

    md("""---
## Stage 1 · Ingest

Turns any input into comparable frames: resized so the long edge is 1024 px, then
filtered for blur (`sharpness < 60`) and exposure. Resizing matters because every
measurement is in **pixels** until stage 4.

`data/input/` accepts a photo, a video, or **a folder of stills of one room** — one
folder is one room, not one dataset, because stages 7-8 deduplicate on the
assumption every frame shows the same place.
""")
    code("""IN_DIR = HERE / 'data' / 'input'
EXT = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.mkv', '.avi', '.m4v'}
if INPUT is None:
    found = sorted(q.name for q in IN_DIR.iterdir()
                   if q.is_dir() or q.suffix.lower() in EXT)
    assert found, f'nothing in {IN_DIR} — put a photo or a folder of stills there'
    INPUT = found[0]
    print(f'using {INPUT!r}   (available: {found})')
src = IN_DIR / INPUT
assert src.exists(), f'not found: {src}'

sampled, frames = P.load_keyframes(src, max_frames=MAX_FRAMES)
print(f'stage 1  {len(frames)}/{len(sampled)} frames kept')
for f in sampled:
    print(f"   t={f['t']:5.1f}s  {f['img'].shape[1]}x{f['img'].shape[0]}  "
          f"sharpness={f['sharp']:6.0f}  brightness={f['bright']:5.1f}  "
          f"{'KEPT' if f['ok'] else 'rejected'}")

frame = frames[0]
plt.figure(figsize=(9, 6))
plt.imshow(cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title(f'stage 1 — the frame we will measure  ({INPUT})'); plt.show()""")

    md("""---
## Load the models

**The slow cell: expect ~6 minutes.** Almost all of it is the detector's first
forward pass on MPS, not the download. Run it once and leave the kernel alive.

`registry.build()` returns an adapter without weights; they load on first use,
which is why the cost appears in stage 2 rather than here.
""")
    code(f"""from poc.models import registry
det = registry.build({cfg.DETECTOR!r}, box_threshold=BOX_TH, text_threshold=TEXT_TH)
dep = registry.build({cfg.DEPTH!r})
seg = registry.build({(seg or 'sam2')!r}) if USE_MASKS else None

for m in (det, dep, seg):
    if m is not None:
        i = m.info
        print(f'  {{i.key:16s}} {{i.display:32s}} {{i.licence:12s}} ship={{i.commercial_ok}}')""")

    md("""---
## Stage 2 · Detection

The only stage that reads the prompt vocabulary. An open-vocabulary detector returns
**matched text**, not a class id, so the label is normalised by `normalise_label()`
in `poc/models/base.py`.

The `reason` column says which way to move `TEXT_TH`:

| reason | meaning | what to do |
|---|---|---|
| *(clean)* | the label was exactly one prompt | nothing — the good case |
| `narrowed` | one prompt found inside longer text | nothing, safe |
| `ambiguous` | several prompts fused — **no single label** | **RAISE** `TEXT_TH` |
| `empty` | nothing cleared the bar | **LOWER** `TEXT_TH` |

An ambiguous box is left unnamed rather than guessed — but its candidates are kept,
and stage 5 searches the **union** of their size classes. That matters: collapsing
ambiguity to nothing once matched a real coffee table to `sofa_3_seat`, 1.4 m³
against a true ~0.3 m³.
""")
    code("""dets = det.detect(frame['img'], PROMPTS)
if seg is not None:
    for d, m in zip(dets, seg.refine(frame['img'], [d.box for d in dets])):
        d.mask = m

print(f'stage 2  {len(dets)} detections  (box_th={BOX_TH}, text_th={TEXT_TH})')
pd.DataFrame([{
    'label': d.label or '(unnamed)',
    'reason': d.label_reason or 'clean',
    'score': round(d.score, 3),
    'raw text from the model': d.label_raw,
    'candidates': list(d.label_candidates),
    'mask': d.mask is not None,
} for d in dets])""")

    md("""---
## Stage 3 · Depth

Predicts an (X, Y, Z) in **metres** per pixel, plus a validity mask and the camera
intrinsics it inferred.

It runs on the **whole frame**, never the crops — these models rely on scene context,
and cropping to a box destroys the geometry they use to reason about scale.

The metres are *claimed*, not measured: roughly 8% error, which is exactly what
stage 4 corrects. A plausible room reads 1-8 m with a 2-3 m vertical span.
""")
    code("""depth = dep.infer(frame['img'])
z = depth.depth[np.isfinite(depth.depth) & depth.mask]
print(f'stage 3  points {depth.points.shape}   valid {depth.mask.mean()*100:.1f}%')
print(f'         depth range {z.min():.2f} - {z.max():.2f} m')
print(f'         intrinsics recovered: {depth.intrinsics is not None}')

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].imshow(cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB)); ax[0].axis('off')
ax[0].set_title('input')
im = ax[1].imshow(np.where(depth.mask, depth.depth, np.nan), cmap='turbo')
ax[1].axis('off'); ax[1].set_title('stage 3 — depth in metres (near = blue)')
fig.colorbar(im, ax=ax[1], shrink=0.8, label='m'); plt.show()""")

    md("""---
## Stage 4 · The scale anchor

**The highest-leverage stage.** A UK internal door is **1.981 m** by standard, so find
the door, see how tall the depth model thinks it is, and divide.

Why it matters more than it looks: depth error is a *single multiplier on the whole
room*, not noise that averages out across objects — and volume goes as length cubed.
A 16% length error becomes a 58% volume error, on every object at once. Correcting it
once fixes everything.

**No door in frame means no anchor**, `SCALE = 1.0`, and the raw depth error flows
straight through to the invoice. That is why `door` recall matters far more than its
small share of the ground truth.
""")
    code("""factor, why = P.door_scale(dets, depth)
SCALE = factor if factor else 1.0
print(f'stage 4  SCALE={SCALE:.4f}')
print(f'         {why}')
if factor:
    measured = 1.981 / factor
    err = (measured / 1.981 - 1) * 100
    print(f'\\n         the model thought the door was {measured:.3f} m; it is 1.981 m')
    print(f'         so every length was {err:+.1f}% off, and every volume '
          f'{((1 + err / 100) ** 3 - 1) * 100:+.0f}%')
    print(f'         SCALE = 1.981 / {measured:.3f} = {factor:.4f}')
else:
    print('\\n         ! No door found. Every dimension below carries the raw error')
    print('           of the depth model: roughly 8% on length, 26% on volume.')""")

    md("""---
## Stage 5 · Measure (Method B)

Three functions per object:

1. **`P.extent`** — takes the depth points inside the box and runs **PCA on the
   horizontal footprint** to find the object's own axes. A sofa at 30° to the camera
   is still 2.2 m long, not 2.5 m of diagonal. Clips at the 2nd/98th percentile so
   one stray pixel cannot stretch it.
2. **`P.resolve_dims`** — a camera never sees the back of a wardrobe, so for
   flat-fronted furniture the depth is unobservable. When it detects that it
   substitutes a class prior and says so in `depth_source`.
3. **`P.allowed_classes`** — how tightly the size-class search is restricted: the
   prompt's own 2-3 classes, or for a fused span the union of its candidates', or
   the whole table if there is genuinely no name.

Read `depth_source`: `observed` means measured, `class_prior` means assumed.
""")
    code("""rows, pairs = [], []          # pairs feed the annotated picture below
for d in dets:
    if d.label == 'door':      # the anchor is a reference, not cargo
        continue
    e = P.extent(depth, d, scale=SCALE)
    if not e:
        print(f'  no usable depth inside the {d.label or "unnamed"} box — skipped')
        continue
    dims, cls, dist, dsrc = P.resolve_dims(e, CLASSES, P.allowed_classes(d, VOCAB))
    row = dict(t=frame['t'], det_label=d.label or '(unnamed)',
               score=round(d.score, 3), **dims,
               mapped_class=cls, map_dist=round(dist, 3), depth_source=dsrc)
    rows.append(row)
    pairs.append((d, row))

print(f'stage 5  {len(rows)} objects measured  (SCALE={SCALE:.4f} applied)')
df = pd.DataFrame(rows)
cols = [c for c in ['det_label', 'w', 'd', 'h', 'bbox_m3', 'mapped_class',
                    'depth_source', 'depth_observed', 'pca_ratio', 'score']
        if c in df.columns]
df[cols]""")

    code("""# Sanity-check against furniture you have actually touched. Without hand-measured
# rooms this is the only quality signal available.
REAL = {'sofa': (0.85, 0.95), 'armchair': (0.75, 0.95),
        'wardrobe': (0.55, 0.65), 'chest of drawers': (0.45, 0.50)}
print('plausibility check — depth (d) is where single-view geometry fails:\\n')
for r in rows:
    hint = REAL.get(r['det_label'])
    flag = ''
    if hint and not (hint[0] * 0.7 <= r['d'] <= hint[1] * 1.3):
        flag = f"  <- d={r['d']:.2f} m, real is {hint[0]}-{hint[1]} m"
    print(f"  {r['det_label']:<18} {r['w']:.2f} x {r['d']:.2f} x {r['h']:.2f} m"
          f"  = {r['bbox_m3']:.3f} m3{flag}")
print('\\nA box contains floor and wall as well as the object, so unmasked depths run')
print('LARGE. Masks fix that but measure only the visible face, so they run SMALL.')
print('Flip USE_MASKS and re-run from stage 2 to see both.')""")

    md("""---
## Stages 7-8 · Deduplicate and total

The same sofa appears in six frames and must be counted **once**. `max` takes the
highest per-class count in any single frame — if one frame saw 2 chairs there are at
least 2. `median` takes the median including zeros, so a class seen in 1 of 8 frames
gets 0 and is dropped: a deliberate false-positive filter.

Two volumes, and **the gap between them is the finding**: `raw` is bounding boxes as
measured and is always larger, because boxes contain background. `class-snapped`
replaces each object with its cube-table class — quote that one.
""")
    code("""inv = P.dedup_measured(rows, DEDUP)
vol_class = P.vol_from_inventory(inv, CLASSES)
vol_raw = sum(r['bbox_m3'] for r in rows)

print(f'stage 7  inventory ({DEDUP}):')
for k, v in sorted(inv.items()):
    print(f"           {v} x {k:<22} @ {CLASSES[k]['cube_m3']:.3f} m3")
print(f'\\nstage 8  class-snapped {vol_class:.3f} m3   raw boxes {vol_raw:.3f} m3')
if vol_class:
    print(f'         the two differ by {abs(vol_raw - vol_class) / vol_class * 100:.0f}%')

prior = sum(1 for r in rows if r['depth_source'] == 'class_prior')
print(f'\\n         depth assumed rather than measured on {prior}/{len(rows)} objects')
if rows and prior / len(rows) > 0.6:
    print('         ! Method B is mostly reading the cube table, so it is no longer')
    print('           an independent check on Method A and their agreement proves little.')""")

    md("""---
## The annotated picture, and the report

Both come from `poc/runner/report.py`, **the same module the CLI uses** — so this is
pixel-for-pixel what `run.py` saves to `results/<run>/frame_000.jpg`.

Colour is provenance, not decoration:

| colour | meaning |
|---|---|
| **green** | measured outright — depth was genuinely visible |
| **amber** | depth was NOT visible; the cube table supplied it |
| **red** | detected but unnamed — size class from geometry alone |
| **blue** | the door: a measuring reference, not cargo |

A picture full of amber and red means the volume rests on assumptions. Worth
labouring: this view is what exposed a 19% over-estimate that looked perfectly
plausible in the JSON.
""")
    code("""doors = [d for d in dets if d.label == 'door']
vis = R.annotate(frame['img'], pairs, doors=doors, scale=SCALE,
                 total_m3=vol_class, title=str(INPUT))
plt.figure(figsize=(13, 9))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title('annotated — identical to what run.py writes to results/'); plt.show()""")

    md("""### The item report

Per-object measurements, then the inventory that would actually be quoted. `run.py`
prints this same table and writes it as `items.csv`, because a surveyor reviews this
in Excel rather than in a JSON viewer.
""")
    code("""print('\\n'.join(R.item_lines(rows, inv, CLASSES)))

out = HERE / 'results' / 'notebook_run'     # this pipeline's own results folder
R.write_csv(rows, inv, CLASSES, out / 'items.csv')
cv2.imwrite(str(out / 'frame_000.jpg'), vis)
print(f'\\nwrote {out}/items.csv and frame_000.jpg')""")

    md("""---
## Now tune it — this is what the notebook is for

The models are warm, so each setting costs ~3 s instead of ~6 minutes. We **mutate**
the thresholds on the warm detector rather than rebuilding it: a fresh adapter
reloads the weights and re-pays the warm-up, which would turn a 20-second sweep into
hours.
""")
    code("""_box, _txt = det.box_th, det.txt_th          # restored at the end
det.txt_th = TEXT_TH
print(f'{"box_th":>7} {"dets":>5} {"named":>6} {"unnamed":>8}   labels found')
for th in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    det.box_th = th
    dd = det.detect(frame['img'], PROMPTS)
    named = [x.label for x in dd if x.label]
    print(f'{th:>7.2f} {len(dd):>5} {len(named):>6} {len(dd) - len(named):>8}   '
          f'{sorted(set(named))}')
det.box_th, det.txt_th = _box, _txt
print('\\nAn object never detected can never be measured, so recall comes first.')
print('A reviewer deletes a false positive in two seconds; they cannot invent')
print('a sofa the model never saw.')""")

    code("""_box, _txt = det.box_th, det.txt_th
det.box_th = BOX_TH
print(f'{"text_th":>8} {"clean":>6} {"narrowed":>9} {"ambiguous":>10} {"empty":>6}')
for th in (0.10, 0.20, 0.25, 0.30, 0.40):
    det.txt_th = th
    dd = det.detect(frame['img'], PROMPTS)
    c = {}
    for x in dd:
        k = x.label_reason or 'clean'
        c[k] = c.get(k, 0) + 1
    print(f"{th:>8.2f} {c.get('clean', 0):>6} {c.get('narrowed', 0):>9} "
          f"{c.get('ambiguous', 0):>10} {c.get('empty', 0):>6}")
det.box_th, det.txt_th = _box, _txt
print('\\nToo LOW  -> prompts fuse into one span -> ambiguous.')
print('Too HIGH -> no token clears the bar     -> empty.')
print('Aim for the value that maximises clean + narrowed.')""")

    md(f"""---
## What this notebook cannot tell you

Every number above is what the pipeline **believes**. Nothing here says whether it is
**right**.

The plausibility check catches the obvious — a 1.74 m deep sofa does not exist — but
it cannot distinguish 3.8 m³ from 5.7 m³ for a real room, and that difference is the
difference between the right van and the wrong one.

For that you need `poc/ground_truth.csv`: 3-5 rooms measured with a tape. Then
`python -m poc.pipelines.{name}.run --input <room> --room <id>` runs stage 9 and
prints bias, absolute error, precision and recall against reality. Until then this is
a demonstration, not a measurement. Gap **A1** in `GAPS.md`.

### Where to go next

- `poc/pipelines/{name}/README.md` — this pipeline
- `PIPELINE.md` — the nine stages with every function and file
- `LEARN.md` — what NMS, IoU and the thresholds actually do
- `python -m poc.pipelines.{name}.run` — all of this in one command
""")
    return C


def write(name: str) -> Path:
    cfg = importlib.import_module(f"poc.pipelines.{name}.config")
    nb = {"cells": build_cells(cfg),
          "metadata": {"kernelspec": {"display_name": "NX Survey POC",
                                      "language": "python", "name": "nx-poc"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / name / "notebook.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a pipeline's step-by-step notebook.")
    ap.add_argument("name", nargs="?", help="pipeline folder name")
    ap.add_argument("--all", action="store_true", help="regenerate every pipeline")
    a = ap.parse_args(argv)

    names = ([q.name for q in sorted(HERE.iterdir())
              if (q / "config.py").is_file()] if a.all else [a.name])
    if not names or names == [None]:
        print("name a pipeline, or --all. Available:")
        for q in sorted(HERE.iterdir()):
            if (q / "config.py").is_file():
                print(f"    {q.name}")
        return 1

    for n in names:
        out = write(n)
        cells = json.loads(out.read_text())["cells"]
        print(f"wrote {out.relative_to(HERE.parents[1])}  "
              f"({len(cells)} cells, "
              f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
