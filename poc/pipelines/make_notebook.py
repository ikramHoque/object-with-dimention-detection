"""
FILE PURPOSE
    Generates a pipeline's step-by-step notebook from its config.py.

HOW TO RUN
    python -m poc.pipelines.make_notebook grounding_dino__moge2__nocls
    python -m poc.pipelines.make_notebook --all

WHY GENERATED RATHER THAN HAND-WRITTEN
    Every pipeline needs the same stages with the same explanations, differing only
    in which models and thresholds they name. Hand-maintaining one notebook per
    pipeline guarantees they diverge: a fix to the stage-4 explanation would land in
    one and not the others, and notebooks are the hardest files to review a diff of.

    Generating them means the explanations have ONE source — this file — and
    regenerating propagates a fix to every pipeline at once.

    Note what is generated and what is not: the notebook's *narration and wiring*
    are generated; the *logic* it calls is imported from poc/runner/pipeline.py.
    Neither is copied per pipeline.

WRITING STYLE FOR THE MARKDOWN CELLS
    Every explanation cell follows the same three-part shape, because a reader
    scanning a long notebook needs to know where to look:

        **What it does.**       plain language, no jargon that is not explained here
        **Why it matters.**     what breaks if this stage is wrong
        **What to look at.**    the specific thing in the output below to check

    Keep sentences short. Prefer a concrete number to an adjective: "a door is
    1.981 m" teaches more than "a known reference height".

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
    cls_key = getattr(cfg, "CLASSIFIER", None)
    no_anchor = getattr(cfg, "NO_ANCHOR", False)
    question = getattr(cfg, "QUESTION", name)

    # ------------------------------------------------------------------ intro
    md(f"""# {name}

**The question this pipeline answers:** {question}

| part | model | job |
|---|---|---|
| detector | `{cfg.DETECTOR}` | find the objects and name them |
| depth | `{cfg.DEPTH}` | say how far away every pixel is |
| segmenter | `{seg or '— none —'}` | {'trim each box down to the object itself' if seg else 'not used here'} |
| classifier | `{cls_key or '— none —'}` | {'a second, independent opinion on each item' if cls_key else 'not used here'} |

## What this notebook is

It is **the same pipeline `run.py` runs**, cut into one cell per step, so you can see
what each step produces before the next one uses it.

It **imports** the real stage functions rather than copying them, so the notebook and
the CLI cannot drift apart. If a number here differs from the CLI's, that is a bug —
please tell me.

## Why bother with a notebook when there is a CLI

Because of one number: **the first time `{cfg.DETECTOR}` looks at an image it takes
about 370 seconds. Every image after that takes about 2.7 seconds.**

The CLI starts a fresh process, so it pays those 370 s *every single time you run it*.
A notebook keeps the models in memory, so you pay it **once** and every re-run after
that is seconds. For trying out settings, that is the difference between an afternoon
and a coffee break.

## Where files go

| | |
|---|---|
| you put pictures in | `data/input/` (this pipeline's own folder) |
| results come out in | `results/` (same folder) |

## The steps, in order

| Step | In plain words | The function that does it |
|---|---|---|
| 1 | Get usable pictures out of the input | `P.load_keyframes` |
| 2 | Find the objects, draw a box round each | detector `.detect()` |
| 3 | Work out how far away everything is | depth `.infer()` |
| 4 | {'**SKIPPED in this pipeline** (that is the experiment)' if no_anchor else '**Correct the scale using a door**'} | `P.door_scale` |
| 5 | Turn each box into width x depth x height | `P.extent`, `P.resolve_dims` |
| 7-8 | Count each item once, add up the volume | `P.dedup_measured`, `P.vol_from_inventory` |

Step 6 — asking a language model to name the items independently — runs in `run.py`,
not here. {'This pipeline has no classifier, so there is nothing missing.' if not cls_key else f'This pipeline uses `{cls_key}`; the notebook covers the measuring half only.'}
(There is no step 9 here either: that is scoring against hand-measured rooms, and
we have none yet.)
""")

    # ------------------------------------------------------------- settings
    md("""---
# The only cell you edit

These values start out copied from `config.py`. Change them **here** to try things
out. Change them in **`config.py`** when you have decided, because `config.py` is
what the CLI reads and what the pipeline officially *is*.
""")
    code(f"""import os
# This has to be set before torch does anything at all. A few operations have no
# Apple-GPU version; without this line they crash instead of quietly using the CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

ONLY       = None              # None = every room. 'bedroom' = one room.
                               # 'bedroom/north.jpg' = one photograph.
FOCUS      = 0                 # which of the selected rooms to walk through in detail
BOX_TH     = {cfg.BOX_TH}               # how sure before we keep a box.  LOWER = find more, keep more junk
TEXT_TH    = {cfg.TEXT_TH}               # how sure of the NAME. step 2 explains which way to move it
USE_MASKS  = {bool(seg)}              # trim boxes down to the object. big effect on step 4
USE_ANCHOR = {not no_anchor}{'              # THIS PIPELINE TURNS THE DOOR CORRECTION OFF ON PURPOSE' if no_anchor else '               # correct the scale using a door of known height'}
MAX_FRAMES = {getattr(cfg, 'MAX_FRAMES', 16)}                 # most frames to take from one video or folder
DEDUP      = {getattr(cfg, 'DEDUP', 'max')!r}            # 'max' believes each sighting; 'median' wants agreement""")

    # ------------------------------------------------------------- imports
    md("""---
# Set up paths, and load the two reference books

Most of this cell is plumbing — it finds this pipeline's folder so the notebook works
whichever directory you started Jupyter from. But three lines in the middle load the
two files the whole pipeline leans on, so they are worth a minute.

## The two books

| file | the question it answers | think of it as |
|---|---|---|
| `poc/detect_vocab.json` | *what am I looking at?* | the shopping list you take to the shop |
| `poc/cube_table.json` | *what is that worth?* | the price list at the till |

You cannot find anything that is not on the shopping list, and you cannot charge for
anything that is not on the price list. Everything below is those two ideas.

## The three variables they become

**`PROMPTS`** — 26 plain English phrases: `sofa`, `wardrobe`, `door`, ... Step 2 hands
this whole list to the detector. This detector is *open-vocabulary*: it has no
built-in idea of what furniture is, you tell it in words every time. So this list
**is** what the pipeline can see. A rug is not "missed" — it was never looked for.

**`CLASSES`** — 42 standard sizes, each with a packed volume. `sofa_3_seat` is
1.4158 m³. This is where the final number on the invoice comes from.

**`VOCAB`** — the shopping list *with* its wiring to the price list:

```
"sofa"  ->  [sofa_2_seat, sofa_3_seat, sofa_sectional]
"chair" ->  [dining_chair, office_chair]
```

This is the part people skip past, and it is the one that does the most work. It is a
**shortlist**: once step 2 says "sofa", step 5 only has to choose between three prices
instead of 42. Worth roughly **18 points of accuracy** — see step 5 for why.

## Two things that will confuse you later if you do not read them now

**The detector never says `sofa_3_seat`.** It only knows the 26 words we gave it, so
the best it can say is `sofa`. Naming and pricing are deliberately two separate jobs,
done by two different steps.

**Packed volume is not width x depth x height.** A 3-seat sofa measures 1.607 m³ as a
box but is priced at 1.416 m³, because a rectangle round an L-shape contains a lot of
air. A double wardrobe goes the other way: 1.440 m³ as a box, 1.699 m³ packed, because
of padding and crating. Movers bill packed volume. That is the entire reason we look
the answer up in a table instead of just multiplying the measurement out.

**Caveat you should know before quoting any number.** `cube_table.json` is marked
`STATUS: PLACEHOLDER`. Published trade cube sheets disagree with each other — the same
item carries different volumes, and "wardrobe" means furniture on one sheet and a
hanging carton on another. Every final number is read off this file, so **no model can
be more accurate than this table is.** Getting NX's own cube sheet is gap A3, and it
is a ceiling on the whole system, not a detail.
""")
    code(f"""import sys, json, re
from datetime import datetime, timezone
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
from poc.runner import pipeline as P      # the maths — the same module the CLI uses
from poc.runner import report as R        # the drawing and the tables — likewise

VOCAB   = json.loads((POC / 'detect_vocab.json').read_text())['prompts']
PROMPTS = list(VOCAB.keys())
CLASSES = json.loads((POC / 'cube_table.json').read_text())['classes']

# One timestamp for everything this run writes. Reports are ADDED, never replaced,
# so you can still read what the previous settings produced.
RUN_STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def safe(s):
    \"\"\"A room name that is safe inside a filename. Real folders get called things
    like 'living room (front)', and those characters are awkward from a shell. The
    readable name is kept inside the CSV and the summary table.\"\"\"
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('._-') or 'room'

print(f'pipeline {{PIPELINE}}')
print(f'run      {{RUN_STAMP}}   (results are timestamped, nothing gets overwritten)')
print(f'folder   {{HERE}}')
print(f'{{len(PROMPTS)}} things we know how to look for, {{len(CLASSES)}} standard sizes')""")

    # ------------------------------------------------------------- stage 1
    md("""---
# Step 1 · Get the pictures

**What it does.** Looks in `data/input/` and lists everything there. **Each item is
treated as a separate room.** Then it loads the frames of *one* of them — the one
`FOCUS` picks — for the detailed walk-through below.

It shrinks every picture so the long edge is 1024 pixels, and throws away frames that
are too blurry (`sharpness < 60`) or too dark or too bright to measure.

**Why it matters.** Everything up to step 4 is measured in **pixels**, so all the
frames have to be the same size or the numbers are not comparable.

**A FOLDER IS A ROOM, and the folder name IS the room name.**

```
data/input/
    bedroom/          <- room "bedroom"
        north.jpg         all three are views of the SAME room
        south.jpg
        window.jpg
    lounge/           <- room "lounge"
    kitchen/          <- room "kitchen"
    hall/             <- room "hall"
```

Four folders means a four-room house. The room name travels with every number the
pipeline produces — into the report, the CSV, the annotated picture and the filenames.

**Why folders and not loose files.** Steps 7-8 count each item once *per room*: a
sofa photographed from three angles is one sofa, not three. The code can only do
that if it knows which photographs belong together, and a folder is how you tell it.

A single loose image still works, as a one-photo room named after the file — handy
when you just want to check one picture.

**You can run three different ways:**

| what you want | how |
|---|---|
| every room | leave `ONLY` as `None` |
| one room | `ONLY = 'bedroom'` |
| one photograph | `ONLY = 'bedroom/north.jpg'` |

**What to look at.** The list of rooms, and whether any frame says `rejected`. Losing
a frame to blur is normal; losing all of them means the input is unusable.
""")
    code("""IN_DIR = HERE / 'data' / 'input'

# P.discover_rooms is the SAME function the CLI uses, so the notebook and the CLI
# always agree on what the rooms are. It is the only place the folder-is-a-room
# rule lives.
ROOMS = P.discover_rooms(IN_DIR)
assert ROOMS, (f'no rooms in {IN_DIR}\\n'
               f'  make one:  mkdir -p {IN_DIR}/bedroom  &&  cp photos/*.jpg '
               f'{IN_DIR}/bedroom/')

if ONLY:
    src = IN_DIR / ONLY
    assert src.exists(), f'ONLY={ONLY!r} not found under {IN_DIR}'
    RUN = [dict(name=P.room_name_for(src, IN_DIR), path=src,
                kind='folder' if src.is_dir() else 'photo', n_images=0)]
else:
    RUN = ROOMS

print(f'{len(ROOMS)} room(s) in data/input:')
for i, r in enumerate(ROOMS):
    what = f"{r['n_images']} photo(s)" if r['kind'] == 'folder' else f"a single {r['kind']}"
    mark = '  <- FOCUS' if (not ONLY and i == FOCUS) else ''
    print(f"  [{i}] {r['name']:<34} {what:<16}{mark}")
if ONLY:
    print(f'\\nONLY is set, so just this: {RUN[0]["name"]}  ({ONLY})')

assert 0 <= FOCUS < len(RUN), f'FOCUS must be 0..{len(RUN) - 1} for this selection'
focus = RUN[FOCUS]
src, ROOM = focus['path'], focus['name']
INPUT = src.name
print(f'\\nwalking through room {ROOM!r} below; every selected room runs further down.')

sampled, frames = P.load_keyframes(src, max_frames=MAX_FRAMES)
print(f'\\nstep 1  kept {len(frames)} of {len(sampled)} frames')
for f in sampled:
    print(f"   t={f['t']:5.1f}s  {f['img'].shape[1]}x{f['img'].shape[0]}  "
          f"sharpness={f['sharp']:6.0f}  brightness={f['bright']:5.1f}  "
          f"{'KEPT' if f['ok'] else 'rejected'}")

frame = frames[0]
plt.figure(figsize=(9, 6))
plt.imshow(cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title(f'step 1 — the frame we are about to measure  (room {ROOM})'); plt.show()""")

    # ------------------------------------------------------------- models
    md("""---
# Load the models

**Expect about 6 minutes, once.** Almost all of it is the detector waking up, not
downloading. Run this cell, go and get a coffee, and then **leave the kernel
running** — everything after this is seconds.

`registry.build()` hands back a wrapper with no weights in it yet; the weights load
the first time you actually use it. That is why the wait shows up in step 2 rather
than here.

**What to look at.** The `ship=` column. `ship=True` means the licence lets NX use it
in the product. `ship=False` means this model is for comparison only.
""")
    code(f"""from poc.models import registry
det = registry.build({cfg.DETECTOR!r}, box_threshold=BOX_TH, text_threshold=TEXT_TH)
dep = registry.build({cfg.DEPTH!r})
seg = registry.build({(seg or 'sam2')!r}) if USE_MASKS else None

for m in (det, dep, seg):
    if m is not None:
        i = m.info
        print(f'  {{i.key:16s}} {{i.display:32s}} {{i.licence:12s}} ship={{i.commercial_ok}}')""")

    # ------------------------------------------------------------- stage 2
    md("""---
# Step 2 · Find the objects

**What it does.** Hands the detector our shopping list — all 26 `PROMPTS`, glued into
one sentence: `"sofa. armchair. coffee table. ... door."` — and gets back a box round
each thing it found, plus a confidence score.

**It names things. It does not price them.** The best it can ever say is `sofa`,
because `sofa` is the word we gave it; it has no idea the price list holds three
different sofas. Choosing *which* sofa is step 5's job, using the measured size.
Keeping naming and pricing apart is deliberate — it means a bad measurement cannot
rename the item, and a bad name cannot invent a measurement.

**The catch that causes most of the confusion here.** This kind of detector does not
return a tidy class number. It returns **whatever words it matched**, which can be a
mangled join of two of our prompts — literally `"sofa wardrobe"` as one string. So
`normalise_label()` in `poc/models/base.py` has to work out what was really meant, and
the `reason` column records what it decided.

| `reason` | what happened | what you should do |
|---|---|---|
| *(clean)* | matched exactly one thing on our list | nothing — this is the good case |
| `narrowed` | our word was inside a longer phrase | nothing, this is safe |
| `ambiguous` | several of our words got fused together | **raise** `TEXT_TH` |
| `empty` | nothing matched well enough | **lower** `TEXT_TH` |
| `unmatched` | the model named something real, but it is **not on our list** | add it to `poc/detect_vocab.json`, or ignore it |

`unmatched` is the one people misread. It is not a threshold problem — the detector
did its job and found a genuine object, we just never asked for that word. On these
living-room photos it returns "rug", which is a real thing in the room and a real
thing on a removal van, but `rug` is not one of our 26 prompts. The box is still
measured; it just has no name to narrow the size-class search with.

**Why we leave ambiguous boxes unnamed instead of guessing.** When a box could be a
sofa or a coffee table, we keep *both* possibilities and let step 5 pick using the
measured size. An earlier version guessed, and matched a real coffee table to
`sofa_3_seat` — 1.4 m³ charged for something that is really about 0.3 m³.

**What to look at.** The `reason` column. Lots of `ambiguous` or `empty` means
`TEXT_TH` is wrong for your pictures — the table above says which way to move it.
""")
    code("""dets = det.detect(frame['img'], PROMPTS)
if seg is not None:
    for d, m in zip(dets, seg.refine(frame['img'], [d.box for d in dets])):
        d.mask = m

print(f'step 2  found {len(dets)} objects  (box_th={BOX_TH}, text_th={TEXT_TH})')
pd.DataFrame([{
    'label': d.label or '(unnamed)',
    'reason': d.label_reason or 'clean',
    'score': round(d.score, 3),
    'raw text from the model': d.label_raw,
    'possible names kept': list(d.label_candidates),
    'has mask': d.mask is not None,
} for d in dets])""")

    # ------------------------------------------------------------- stage 3
    md("""---
# Step 3 · How far away is everything

**What it does.** Gives every pixel an (X, Y, Z) position in **metres**, from a single
ordinary photo. No special camera, no laser.

**Two things worth knowing.**

It runs on the **whole picture**, never on the cropped boxes. These models judge
distance from context — the size of the room, where the floor is — and cropping to a
box throws that context away.

The metres are a good *guess*, not a measurement: typically about **8% out**. Fixing
that is exactly what step 4 is for.

**What to look at.** The depth range. A normal room reads roughly 1-8 m. If it reads
0.2-1.5 m, the model has misjudged the whole scene and every volume below will be
wrong together.
""")
    code("""depth = dep.infer(frame['img'])
z = depth.depth[np.isfinite(depth.depth) & depth.mask]
print(f'step 3  a 3D point per pixel: {depth.points.shape}')
print(f'        {depth.mask.mean()*100:.1f}% of pixels got a usable distance')
print(f'        nearest {z.min():.2f} m, furthest {z.max():.2f} m')
print(f'        worked out the camera lens too: {depth.intrinsics is not None}')

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].imshow(cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB)); ax[0].axis('off')
ax[0].set_title('what the camera saw')
im = ax[1].imshow(np.where(depth.mask, depth.depth, np.nan), cmap='turbo')
ax[1].axis('off'); ax[1].set_title('step 3 — distance in metres (blue = close)')
fig.colorbar(im, ax=ax[1], shrink=0.8, label='m'); plt.show()""")

    # ------------------------------------------------------------- stage 4
    md(f"""---
# Step 4 · {'The door correction — TURNED OFF in this pipeline' if no_anchor else 'Correct the scale using a door'}

**What it does.** A UK internal door is **1.981 m** tall, by standard. So: find the
door, ask the depth model how tall *it* thinks the door is, and divide. If the model
says 1.85 m, everything it measured is 7% too small — so multiply every length by
1.981 / 1.85 and the whole room is fixed at once.

**Why this is the highest-value step in the pipeline.** The depth model's error is
**one multiplier applied to the entire room**, not random noise. That means it does
*not* average out over many objects — every object is wrong in the same direction.
Ten objects do not cancel each other out; they all lean the same way.

And volume is length **cubed**, so the error is tripled. Here is what a wrong door
actually costs, with the real arithmetic:

| depth model thinks the door is | so `SCALE` = 1.981 / that | every volume in the room moves by |
|---|---|---|
| 1.850 m | 1.0708 | **+22.8%** |
| 1.900 m | 1.0426 | +13.3% |
| 1.981 m | 1.0000 | 0.0% — already right |
| 2.100 m | 0.9433 | -16.1% |

A **7%** mistake on one length became a **23%** mistake on the bill. Read that table
once and the rest of the pipeline's design makes sense.

One door fixes all of it, in one multiplication. This is also why finding doors
matters far more than their small share of the objects in a room would suggest — the
door is not cargo, it is the ruler.

{'**This pipeline deliberately skips it.** `USE_ANCHOR = False`, so `SCALE` stays at 1.00 and the raw depth error flows straight through. That is the point: comparing this run against the pipeline that keeps the anchor shows you exactly what the door is worth in cubic metres.' if no_anchor else '**If there is no door in the picture** there is nothing to calibrate against, `SCALE` stays 1.00, and that 8% depth error goes straight onto the invoice.'}

**What to look at.** `SCALE`. Near 1.00 means the depth model was already about right.
Far from it means the door just saved you from a large error.
""")
    code("""if USE_ANCHOR:
    factor, why = P.door_scale(dets, depth)
else:
    factor, why = None, 'anchor disabled in this pipeline (USE_ANCHOR = False)'
SCALE = factor if factor else 1.0
print(f'step 4  SCALE = {SCALE:.4f}')
print(f'        {why}')
if factor:
    measured = 1.981 / factor
    err = (measured / 1.981 - 1) * 100
    print(f'\\n        the model thought the door was {measured:.3f} m tall.')
    print(f'        A UK internal door is 1.981 m.')
    print(f'        So every length was {err:+.1f}% out, which makes every volume '
          f'{((1 + err / 100) ** 3 - 1) * 100:+.0f}% out.')
    print(f'        Correction: 1.981 / {measured:.3f} = {factor:.4f}')
else:
    print('\\n        ! No correction applied. Every measurement below carries the')
    print('          depth model\\'s raw error: roughly 8% on length, 26% on volume.')""")

    # ------------------------------------------------------------- stage 5
    md("""---
# Step 5 · Turn each box into a size, then into a price

This is where the two reference books finally meet. Easiest way in is to follow one
sofa the whole way through. Every number below is real — it comes from running these
functions.

## Worked example: one sofa

**5.1 `P.extent` measures the box.** Takes the 3D points inside the box and finds the
object's *own* directions rather than the camera's. A sofa sitting at 30° to the
camera is still 2.2 m long; measuring along the camera's axes would call it 2.5 m.
(The technique is PCA on the footprint — it finds the direction the object is longest
in.) It throws away the most extreme 2% of points so one stray pixel cannot stretch
the answer.

```
measured:  w 2.00   d 0.88   h 0.83     ->  box volume 1.461 m3
```

**5.2 `P.allowed_classes` builds the shortlist.** Step 2 said the label was `sofa`, so:

```
VOCAB['sofa']  ->  [sofa_2_seat, sofa_3_seat, sofa_sectional]
```

Three candidates instead of 42.

**5.3 `P.nearest_class` picks the closest.** It compares the measurement against each
candidate's typical size and keeps the smallest difference:

| candidate | typical size | distance |
|---|---|---|
| `sofa_2_seat` | 1.6 x 0.9 x 0.85 | 0.198 |
| **`sofa_3_seat`** | **2.1 x 0.9 x 0.85** | **0.043**  <- winner |
| `sofa_sectional` | 2.8 x 1.7 x 0.85 | 0.339 |

**Charge: 1.4158 m3** — read off the price list.

Notice the charge (1.416) is *not* the measured box (1.461). We measured in order to
**identify** the item; the price then comes from the table. That is on purpose, and it
is what a mover actually bills.

## Why the shortlist exists — the bit that is easy to miss

With a *good* measurement the shortlist changes nothing: searching all 42 also picks
`sofa_3_seat`. **It earns its keep when the measurement is wrong** — which, without
masks, it usually is, because the box round a sofa also contains floor and wall.

Real case from our own masked-vs-unmasked run: that same sofa's front-to-back depth
read **1.61 m** instead of ~0.88 m.

| | winner | charge |
|---|---|---|
| shortlist **on** | `sofa_sectional` | 2.265 m3 |
| shortlist **off** | **`bed_king_frame`** | 1.416 m3 |

Look carefully. Without the shortlist the *number* happens to land closer to the
truth — but it has put **a king-size bed in the living room**. A surveyor reading that
cube sheet stops trusting the entire document.

So the shortlist does not make the measurement better. **It stops a bad measurement
from becoming a nonsense item.** A wrong-size sofa is a recoverable error; furniture
that does not exist is not.

## `P.resolve_dims` — being honest about what the camera cannot see

A camera never sees the back of a wardrobe. For anything flat against a wall the
front-to-back depth is simply not in the photograph:

```
measured:  w 1.20   d 0.11   h 2.00     ->  0.264 m3
```

0.11 m deep is obviously wrong, but the code cannot measure what is not there. So it
matches on **width and height only**, then borrows that class's standard depth:

```
after:     w 1.20   d 0.60   h 2.00     ->  1.440 m3
class wardrobe_double    depth_source = class_prior    charge 1.699 m3
```

The important part is that it **writes `class_prior` in the output** instead of
passing 0.60 m off as a measurement.

**What to look at.** The `depth_source` column, every single time:

| value | meaning |
|---|---|
| `observed` | genuinely measured from the picture |
| `class_prior` | **assumed** from a standard furniture size |

If most rows say `class_prior`, this pipeline is mostly *looking things up*, not
measuring your room — and you need to know that before claiming measurement accuracy.
""")
    code("""rows, pairs = [], []          # `pairs` feeds the annotated picture further down
for d in dets:
    if d.label == 'door':      # the door is our ruler, not something being moved
        continue
    e = P.extent(depth, d, scale=SCALE)
    if not e:
        print(f'  no usable depth inside the {d.label or "unnamed"} box — skipped')
        continue
    dims, cls, dist, dsrc = P.resolve_dims(e, CLASSES, P.allowed_classes(d, VOCAB))
    # det_label stays '' when we could not name it — NOT the string '(unnamed)'.
    # report.py decides the box colour by asking whether this is empty, so a
    # placeholder here would draw an unidentified object in green, as if it had
    # been named and measured. Display turns '' into '(unnamed)'; the data does not.
    row = dict(t=frame['t'], det_label=d.label,
               score=round(d.score, 3), **dims,
               mapped_class=cls, map_dist=round(dist, 3), depth_source=dsrc)
    rows.append(row)
    pairs.append((d, row))

print(f'step 5  measured {len(rows)} objects  (SCALE={SCALE:.4f} applied to all)')
df = pd.DataFrame(rows)
if not df.empty:
    df['det_label'] = df['det_label'].replace('', '(unnamed)')
cols = [c for c in ['det_label', 'w', 'd', 'h', 'bbox_m3', 'mapped_class',
                    'depth_source', 'depth_observed', 'pca_ratio', 'score']
        if c in df.columns]
df[cols]""")

    code("""# A rough reality check against furniture you have actually touched. Until we have
# hand-measured rooms (gap A1), this is the only quality signal we have at all.
REAL = {'sofa': (0.85, 0.95), 'armchair': (0.75, 0.95),
        'wardrobe': (0.55, 0.65), 'chest of drawers': (0.45, 0.50)}
print('Reality check. `d` is front-to-back, which is where single photos struggle:\\n')
for r in rows:
    hint = REAL.get(r['det_label'])
    flag = ''
    if hint and not (hint[0] * 0.7 <= r['d'] <= hint[1] * 1.3):
        flag = f"  <- says {r['d']:.2f} m, real ones are {hint[0]}-{hint[1]} m"
    print(f"  {(r['det_label'] or '(unnamed)'):<18} "
          f"{r['w']:.2f} x {r['d']:.2f} x {r['h']:.2f} m"
          f"  = {r['bbox_m3']:.3f} m3{flag}")
print('\\nA box round a sofa also contains some floor and some wall, so measuring')
print('the whole box reads too BIG. Masks cut that away but then only see the')
print('front face, so they read too SMALL. Flip USE_MASKS and re-run from step 2')
print('to see both, if this pipeline has a segmenter.')""")

    # ------------------------------------------------------------- stage 7-8
    md("""---
# Steps 7-8 · Count each thing once, then add it up

**What it does.** The same sofa shows up in six photos of the same room and must be
charged for **once**.

| `DEDUP` | how it counts | when to use it |
|---|---|---|
| `'max'` | the most seen in any single frame | trusts every sighting — if one frame saw 2 chairs, there are at least 2 |
| `'median'` | the middle count across all frames, zeros included | something seen in 1 frame out of 8 gets dropped as a false alarm |

Then two totals come out, and **the difference between them is the interesting part**:

- **raw boxes** — adding up the boxes exactly as measured. Always the bigger number,
  because a box round a sofa contains floor and wall too.
- **class-snapped** — each object replaced by its standard furniture size. **This is
  the one you would quote**, because it is what a surveyor's cube sheet would say.

**What to look at.** The last line: how many objects had their depth *assumed* rather
than measured. Past about 60% the measuring half of the pipeline is really just
reading the standard-size table, so it has stopped being an independent check on
anything.
""")
    code("""inv = P.dedup_measured(rows, DEDUP)
vol_class = P.vol_from_inventory(inv, CLASSES)
vol_raw = sum(r['bbox_m3'] for r in rows)

print(f'step 7  the inventory (counting method: {DEDUP}):')
for k, v in sorted(inv.items()):
    print(f"          {v} x {k:<22} @ {CLASSES[k]['cube_m3']:.3f} m3 each")
print(f'\\nstep 8  quotable total   {vol_class:.3f} m3')
print(f'        raw boxes        {vol_raw:.3f} m3')
if vol_class:
    print(f'        the two differ by {abs(vol_raw - vol_class) / vol_class * 100:.0f}%')

prior = sum(1 for r in rows if r['depth_source'] == 'class_prior')
print(f'\\n        depth was ASSUMED, not measured, on {prior}/{len(rows)} objects')
if rows and prior / len(rows) > 0.6:
    print('        ! Most depths came from the lookup table, so this half of the')
    print('          pipeline is no longer an independent check on the other half.')""")

    # ------------------------------------------------------------- report
    md("""---
# The annotated picture

**What it does.** Draws the boxes on the photo — using `poc/runner/report.py`, **the
same code the CLI uses**, so this is pixel-for-pixel what `run.py` saves into
`results/`.

**The colours are not decoration. They tell you where each number came from:**

| colour | meaning |
|---|---|
| **green** | properly measured — the depth was really visible |
| **amber** | depth was *not* visible; a standard size was assumed |
| **red** | found, but we do not know what it is; size guessed from shape alone |
| **blue** | the door — our ruler, not cargo |

**Why this picture is worth more than the JSON.** A picture full of amber and red
means the total rests on assumptions. This view is what caught a 19% over-estimate
that looked completely reasonable as numbers on a screen.
""")
    code("""doors = [d for d in dets if d.label == 'door']
vis = R.annotate(frame['img'], pairs, doors=doors, scale=SCALE,
                 total_m3=vol_class, title=f'room: {ROOM}')
plt.figure(figsize=(13, 9))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title('identical to what run.py writes into results/'); plt.show()""")

    md("""### The item report

Every object measured, then the inventory that would actually be quoted. `run.py`
prints this same table and saves it as `items.csv`, because a surveyor checks this
sort of thing in Excel, not in a JSON viewer.
""")
    code("""print('\\n'.join(R.item_lines(rows, inv, CLASSES)))

# Room name and timestamp in the path, so a re-run ADDS a report next to the last
# one instead of overwriting it. You can compare two settings after the fact.
out = HERE / 'results' / f'notebook__{RUN_STAMP}' / 'walkthrough' / safe(ROOM)
R.write_csv(rows, inv, CLASSES, out / 'items.csv')
cv2.imwrite(str(out / 'frame_000.jpg'), vis)
print(f'\\nsaved to {out.relative_to(HERE)}/')
print('      items.csv, frame_000.jpg')
print('\\nThat is the ONE frame walked through above. The section further down writes')
print(f"every frame of every room to results/notebook__{RUN_STAMP}/<room>/.")""")

    # ------------------------------------------------------------- all rooms
    md("""---
# Now every room in `data/input/`

Everything above walked through **one** room, and only its **first frame**, so you
could see each step. This section does the **whole job**: every room, every frame,
exactly as `run.py --all` does it.

**What it does.** `run_room()` below calls the very same `P.*` functions you just
stepped through — nothing is reimplemented. It loops over the frames of one room,
then over the rooms.

**Why it is fast now.** The models are already in memory. The 370-second wait was
paid once, up in the load cell. Each extra frame costs about 3 seconds.

**Each room stays separate.** One inventory and one total per room, never merged —
they are different rooms, and merging them would produce a total for a house that
does not exist.

**What to look at.** The `anchored` column in the summary. Any room with `anchored =
False` had no door in shot, so its volume carries the full uncorrected depth error
and is not comparable with the others.
""")
    code("""# P.measure_room IS steps 1-5. The same function the CLI calls — not a copy of
# it — so a number here cannot differ from a number `run.py` prints. Everything
# you stepped through above happens inside it.
def run_room(room):
    M = P.measure_room(room['path'], det, dep, seg,
                       prompts=PROMPTS, classes=CLASSES, vocab=VOCAB,
                       max_frames=MAX_FRAMES, use_anchor=USE_ANCHOR)
    inv = P.dedup_measured(M['rows'], DEDUP)                        # step 7
    return dict(room=room['name'], **M, inventory=inv,              # step 8
                vol_class=P.vol_from_inventory(inv, CLASSES),
                vol_raw=sum(r['bbox_m3'] for r in M['rows']))

results = {}
for i, r in enumerate(RUN, 1):
    print(f"[{i}/{len(RUN)}] {r['name']:<28}", end=' ', flush=True)
    try:
        res = results[r['name']] = run_room(r)
        print(f"{len(res['frames'])} frame(s), {len(res['rows']):>2} object(s), "
              f"{res['vol_class']:>7.3f} m3   scale={res['scale']:.3f}")
    except Exception as exc:            # one bad room must not lose the others
        print(f'FAILED: {type(exc).__name__}: {exc}')""")

    code("""# One row per room. This is the table you would take to a review.
summary = pd.DataFrame([{
    'room': r['room'],
    'frames': f"{len(r['frames'])}/{len(r['sampled'])}",
    'objects': len(r['rows']),
    'anchored': r['scale'] != 1.0,
    'scale': round(r['scale'], 4),
    'quotable_m3': round(r['vol_class'], 3),
    'raw_boxes_m3': round(r['vol_raw'], 3),
    'depth_assumed': f"{r['prior']}/{len(r['rows'])}",
    'unnamed': r['lab_missing'],
    'items': ', '.join(f'{v}x{k}' for k, v in sorted(r['inventory'].items())) or '(none)',
} for r in results.values()])
display(summary)

if len(results) > 1:
    print(f"\\n{len(results)} rooms, {summary['quotable_m3'].sum():.3f} m3 altogether.")
    print('That sum is a house total only if these rooms are one job.')
un = [r['room'] for r in results.values() if r['scale'] == 1.0]
if un:
    print(f"\\n! No door found in: {', '.join(un)}")
    print("  Those rooms kept SCALE=1.00, so they carry the depth model's raw error")
    print('  (~26% on volume) and are not comparable with anchored rooms.')""")

    code("""# Write it all out, under this run's timestamp. Nothing is overwritten: run again
# with different settings and you get a second folder you can compare against.
out_all = HERE / 'results' / f'notebook__{RUN_STAMP}'
out_all.mkdir(parents=True, exist_ok=True)
summary.insert(0, 'pipeline', PIPELINE)
summary.insert(1, 'run', RUN_STAMP)
summary.to_csv(out_all / 'summary.csv', index=False)

for r in results.values():
    d = out_all / safe(r['room'])
    R.write_csv(r['rows'], r['inventory'], CLASSES, d / 'items.csv')
    imgs = R.write_visuals(r['frames'], d, scale=r['scale'], total_m3=r['vol_class'])
    (d / 'room.txt').write_text(
        f"room       {r['room']}\\npipeline   {PIPELINE}\\nrun        {RUN_STAMP}\\n"
        f"frames     {len(r['frames'])}/{len(r['sampled'])}\\n"
        f"scale      {r['scale']:.4f}  ({r['scale_src']})\\n"
        f"quotable   {r['vol_class']:.3f} m3\\nraw boxes  {r['vol_raw']:.3f} m3\\n")
    print(f"  {r['room']:<28} -> {d.relative_to(HERE)}/  "
          f"({len(imgs)} frame(s) + items.csv + room.txt)")
print(f"\\nsummary.csv covering all {len(results)} room(s): "
      f"{(out_all / 'summary.csv').relative_to(HERE)}")""")

    code("""# The annotated first frame of every room, side by side. The quickest way to see
# that a room went wrong — you spot it before reading a single number.
n = len(results)
if n:
    cols = min(n, 3)
    rowsn = (n + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(6.5 * cols, 5 * rowsn), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, r in zip(axes.flat, results.values()):
        f = r['frames'][0]
        v = R.annotate(f['img'], f['pairs'],
                       doors=[d for d in f['dets'] if d.label == 'door'],
                       scale=r['scale'], total_m3=r['vol_class'],
                       title=f"room: {r['room']}")
        ax.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{r['room']}  —  {r['vol_class']:.3f} m3"
                     f"{'' if r['scale'] != 1.0 else '   (NO DOOR)'}", fontsize=11)
    plt.tight_layout(); plt.show()""")

    md("""### Does the walk-through agree with the shared function?

It has to. The cells above stepped through one room by hand; `P.measure_room` did the
same room inside `run_room()`. If those two ever disagree, everything the notebook
claims about matching the CLI is false — so rather than trust it, the next cell
checks it.
""")
    code("""chk = results.get(ROOM)
if chk is None:
    print(f'room {ROOM!r} was not in this run — nothing to compare')
else:
    mine = sorted((r['det_label'], round(r['bbox_m3'], 4)) for r in rows)
    thrs = sorted((r['det_label'], round(r['bbox_m3'], 4))
                  for r in chk['rows'] if r['t'] == frame['t'])
    assert abs(chk['scale'] - SCALE) < 1e-9, (
        f"SCALE differs: walk-through {SCALE}, measure_room {chk['scale']}")
    assert mine == thrs, (
        'the hand-stepped cells and P.measure_room disagree!\\n'
        f'  notebook:     {mine}\\n  measure_room: {thrs}')
    print(f'agreed: SCALE {SCALE:.4f} and {len(mine)} identical measurements.')
    print('The notebook and the CLI are running the same arithmetic.')""")

    # ------------------------------------------------------------- tuning
    md("""---
# Now tune it — this is the real reason for the notebook

The models are warm, so each setting costs about 3 seconds instead of 6 minutes.

Note what these cells do **not** do: they do not rebuild the detector. They change
the threshold on the warm one and put it back afterwards. Rebuilding would reload the
weights and re-pay the warm-up, turning a 20-second sweep into hours.
""")
    code("""_box, _txt = det.box_th, det.txt_th          # put back at the end
det.txt_th = TEXT_TH
print('How many objects do we find as we lower the bar?\\n')
print(f'{"box_th":>7} {"found":>6} {"named":>6} {"unnamed":>8}   what it named')
for th in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    det.box_th = th
    dd = det.detect(frame['img'], PROMPTS)
    named = [x.label for x in dd if x.label]
    print(f'{th:>7.2f} {len(dd):>6} {len(named):>6} {len(dd) - len(named):>8}   '
          f'{sorted(set(named))}')
det.box_th, det.txt_th = _box, _txt
print('\\nWhich way to go: something never found can never be measured, so')
print('finding more matters more than finding less junk. A surveyor deletes a')
print('false alarm in two seconds. They cannot invent a sofa nobody photographed.')""")

    code("""_box, _txt = det.box_th, det.txt_th
det.box_th = BOX_TH
print('And how good are the NAMES as we move the other threshold?\\n')
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
print('\\nToo LOW  -> our prompts fuse into one blob  -> ambiguous')
print('Too HIGH -> no word is convincing enough     -> empty')
print('Pick the value with the most clean + narrowed.')""")

    # ------------------------------------------------------------- honesty
    md(f"""---
# What this notebook cannot tell you

Every number above is what the pipeline **believes**. Nothing here tells you whether
it is **right**.

The reality check catches the obvious — no sofa is 1.74 m front-to-back. It cannot
tell you whether a real room is 3.8 m³ or 5.7 m³. And that particular difference is
the difference between sending the right van and the wrong one.

**What would fix it:** `poc/ground_truth.csv`, holding 3-5 rooms measured with a tape
measure. Then:

```
python -m poc.pipelines.{name}.run --input <room> --room <id>
```

runs the scoring step and prints how far off it was, in both directions. Until that
file exists, this is a **demonstration, not a measurement**. It is gap **A1** in
`GAPS.md`, and it is the single most valuable thing anyone could add.

### Doing this from the command line instead

```
python -m poc.pipelines.{name}.run --input lounge.jpg   # one room
python -m poc.pipelines.{name}.run --all                # every room in data/input
```

Same code, same numbers — but it pays the 370-second warm-up on every invocation,
which is why tuning belongs here and production runs belong there.

### Where to read more

| file | what is in it |
|---|---|
| `DIAGRAM.md` | this pipeline as a diagram — the one to show your team lead |
| `README.md` | what this pipeline is for |
| `PIPELINE.md` | all the steps, with every function and file named |
| `LEARN.md` | what the thresholds, NMS and IoU actually mean |
| `GAPS.md` | what we know is still missing |
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
