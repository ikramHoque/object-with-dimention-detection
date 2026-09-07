# The Pipeline, Method by Method

> **FILE PURPOSE** — What actually happens to an image, in order, with the function
> and file that does each step, and where every input and output lives. Written
> because the metrics tools and the pipeline are different things, and it was not
> obvious which one produces the product.
>
> Every number in this file comes from one real run on 7 Sept 2026, not an estimate.

---

## 0. The distinction that causes all the confusion

There are **two different tools**. They answer different questions and only one of
them produces the product.

| | `run_dataset_eval` | `run_combination` |
|---|---|---|
| Question | "does the detector find the box?" | "how much space will this shipment take?" |
| Input | the HomeObjects dataset | **your** photo / video / folder |
| Output | precision, recall, F1 **printed** | a **JSON file** with objects, dimensions, volume |
| Computes dimensions? | **no** | yes |
| Computes volume? | **no** | yes |
| Needs ground truth? | uses the dataset's boxes | only for stage 9 scoring |
| Why it exists | tune the detector's knobs | the actual deliverable |

**If you only ever ran `run_dataset_eval`, you have never seen the pipeline's output.**
That tool stops after detection. It cannot tell you output quality, because it never
computes an output — only whether boxes landed in the right places.

```
run_dataset_eval   image ──► detect ──► compare to dataset boxes ──► precision/recall
run_combination    image ──► detect ──► depth ──► scale ──► measure ──► volume ──► JSON
                                        └──────── the part eval never runs ────────┘
```

---

## 1. The nine stages

One command runs all of them:

```bash
python -m poc.runner.run_combination --input demo_room.jpg \
    --detector grounding_dino --depth moge2 --classifier none
```

Everything below lives in **`poc/runner/run_combination.py`** (the orchestrator, which
prints the `stage N` lines you see) and **`poc/runner/pipeline.py`** (the maths). The
orchestrator holds no cleverness; every real calculation is a named function you can
read on its own.

Stages 1-5 for one room are themselves a single shared function, `measure_room()`
(`poc/runner/pipeline.py:418`), called by both the CLI and every pipeline notebook.
That loop used to be written out twice, and the two copies had already drifted apart
— the notebook wrote `det_label='(unnamed)'` where the CLI wrote `''`, which broke the
colour coding: `report._colour()` treats a non-empty label as named, so boxes the
model could NOT identify were drawn green instead of red. Every notebook now ends with
a cell that asserts its hand-stepped result equals `measure_room()`'s, so a future
drift fails loudly instead of quietly misleading a reader.

### Stage 1 · Ingest — turn any input into comparable frames

| | |
|---|---|
| Functions | `discover_rooms()` `:58` (a folder is a room), `room_name_for()` `:105`, `load_keyframes()` `:131` |
| In | `data/input/` — one image, one video, or a folder of stills |
| Out | `[{t, img, sharp, bright, ok}]` — frames, resized so the long edge is 1024 px |
| Printed | `room bedroom` then `stage 1  3/3 frames kept` |

**A SUB-FOLDER IS A ROOM, and its folder name IS the room name.**

```
poc/pipelines/<name>/data/input/
    bedroom/          -> room "bedroom"   every photo inside is a view of it
        north.jpg
        south.jpg
    lounge/           -> room "lounge"
    kitchen/          -> room "kitchen"
    hall/             -> room "hall"
    quick.jpg         -> room "quick"     a loose file is a one-photo room
```

Four folders is a four-room house. The room name is carried into the JSON, the CSV,
the annotated frame and every output filename, so a report always says which room it
describes.

**Why folders rather than loose files.** Stages 7-8 count each item once *per room*: a
sofa photographed from three angles is one sofa. That is only possible if the code
knows which photographs belong together, and a folder is how you say so. Eight loose
photos of eight different rooms would otherwise merge into one inventory for a house
that does not exist.

`room_name_for()` resolves any `--input` to its room, so `--input lounge/north.jpg`
files its results under **lounge**, not "north".

Three ways to select what runs:

| | |
|---|---|
| every room, one process | `--all` (the models load once — see below) |
| one room | `--input lounge` |
| one photograph | `--input lounge/north.jpg` |

Video is sampled once per second. Blurry (`sharpness < 60`) and badly exposed frames
are dropped, so garbage never reaches the models. `3/3 kept` means three frames in,
three survived; `12/16` would mean four were rejected.

**Why resize:** every downstream number is in pixels until stage 4, so frames must be
a consistent size or the scale factor means nothing.

**Why `--all` matters.** Grounding DINO's first inference in a process costs ~370 s
and every one after ~2.7 s (`registry.build` caches adapters per process). Five rooms
in one process is one warm-up; five separate commands is five.

### Stage 2 · Detection — find and name objects

| | |
|---|---|
| Function | the detector adapter's `.detect()`, e.g. `poc/models/detect_grounding_dino.py` |
| In | a frame + the 26 prompts from `poc/detect_vocab.json` |
| Out | `[Detection(label, score, box, label_raw, label_reason)]` |
| Printed | `stage 2  7 detections` and a `labels:` breakdown |

This is the only stage that reads your prompt vocabulary. An open-vocabulary detector
returns **matched text**, not a class id, so the label needs normalising — see
`normalise_label()` in `poc/models/base.py`. The `labels:` line tells you which way
your `text_threshold` is wrong:

```
labels: 2 ambiguous, 4 clean, 1 narrowed (of 7)
! 2/7 spans fused several prompts: text_threshold is too LOW. RAISE it.
```

- **clean** — the label was exactly one prompt. What you want.
- **narrowed** — one prompt found inside longer text. Safe.
- **ambiguous** — several prompts fused; we return **no label** rather than guess.
- **empty** — nothing cleared `text_threshold`. LOWER it.

### Stage 3 · Depth — how far away is every pixel

| | |
|---|---|
| Function | the depth adapter's `.infer()`, e.g. `poc/models/depth_moge2.py` |
| In | the whole frame (not the boxes) |
| Out | `DepthResult(points, depth, mask, intrinsics)` — an (X,Y,Z) in metres per pixel |
| Printed | `stage 3  depth done` |

Runs on the full image because these models need scene context; cropping to a box
destroys the geometry they rely on. The output is *claimed* metric — MoGe-2 predicts
real metres — but with roughly 8% error, which stage 4 exists to correct.

### Stage 4 · Scale anchor — fix the one error that matters most

| | |
|---|---|
| Function | `door_scale()` — `poc/runner/pipeline.py:197` |
| In | this frame's detections + depth |
| Out | one multiplier, e.g. `0.8589` |
| Printed | `stage 4  SCALE=0.8589  [door anchor, median of 1 frames]` |

A UK internal door is **1.981 m** by standard. So: find the door, measure how tall the
depth model *thinks* it is, and divide. In the real run the model claimed **2.307 m**,
so everything was 16% too big and the correction is `1.981 / 2.307 = 0.8589`.

**Why this stage earns its place.** Depth error is a single multiplier on the whole
room, not noise that averages out across objects. And volume goes as length cubed, so

```
2.307 / 1.981 = 1.164  ->  16.4% too long
1.164³        = 1.579  ->  58% too much volume
```

One 16% mistake becomes a 58% over-quote. Correcting it once fixes every object at
the same time. This is why `door` recall matters far more than its 14 ground-truth
boxes suggest — no door, no anchor, `SCALE=1.0`, and the 16% flows straight through.

### Stage 5 · Measure — pixels to metres to cubic metres (Method B)

| | |
|---|---|
| Functions | `extent()` `:224`, `nearest_class()` `:302`, `resolve_dims()` `:324` |
| In | box + depth + scale |
| Out | one row per object: `w, d, h, bbox_m3, mapped_class, depth_source` |
| Printed | `stage 5  6 objects measured` + depth-observability line |

Three steps per object:

1. **`extent()`** takes the depth points inside the box, runs **PCA on the horizontal
   footprint** to find the object's own axes (a sofa at 30° to the camera is still
   2.2 m long, not 2.5 m of diagonal), and clips at the 2nd/98th percentile so one
   stray pixel cannot stretch it.
2. **`resolve_dims()`** handles the single-view limit: a camera never sees the back of
   a wardrobe, so for flat-fronted furniture the depth extent is unobservable. It
   detects that (`depth_observed=False`) and substitutes a class prior. The
   `depth_source` column says `observed` or `class_prior` per object, so you always
   know which numbers were measured and which were assumed.
3. **`nearest_class()`** matches the measured w/d/h to the closest of 42 entries in
   `poc/cube_table.json`. If the label is known it searches only that label's 2–3
   candidates; **if the label is unnamed it searches all 42 by geometry**, which is
   why an unnamed box still produces a usable answer.

### Stage 6 · Recognise (Method A) — the independent second opinion

| | |
|---|---|
| Function | `poc/models/classify_claude.py` |
| In | the frame image (not the boxes) |
| Out | `[{size_class, count, confidence}]` per frame |
| Printed | `stage 6  N frames classified` |
| Skipped by | `--classifier none` — no API key needed, and no cost |

Method A never measures anything. It names objects and reads their volume from the
cube table. Method B measures geometry. They fail differently on purpose: when they
agree you can trust the number; when they diverge you have found a room that needs a
human. That is the whole reason for running both.

### Stages 7–8 · Deduplicate and total

| | |
|---|---|
| Functions | `dedup_counts()` `:368`, `dedup_measured()` `:391`, `vol_from_inventory()` `:411` |
| In | all rows from all frames |
| Out | one inventory + three volume figures |
| Printed | `stage 8  A=0.0 m3   B(class)=5.748 m3   B(raw)=8.795 m3` |

The same sofa appears in six frames and must be counted **once**. `--dedup max` takes
the highest per-class count seen in any single frame — if one frame saw 2 chairs then
there are at least 2. `--dedup median` takes the median **including zeros**, so a class
seen in only 1 of 8 frames gets a median of 0 and is dropped entirely. That is
deliberately a false-positive filter: `max` trusts every sighting, `median` demands
that a majority of frames agree.

Three volumes, and the gap between them is the finding:

- **`recognised_m3`** — Method A, cube table. `0.0` here because `--classifier none`.
- **`measured_class_m3`** — Method B snapped to cube-table classes. **The one to quote.**
- **`measured_raw_m3`** — raw bounding boxes, no snapping. Always the largest, because
  a box around a sofa also contains floor and wall.

`8.795` raw vs `5.748` snapped is a **53% gap**, and that gap is background pixels
inside the boxes. Which leads directly to the next section.

### Stage 9 · Score — only if ground truth exists

| | |
|---|---|
| Function | `score_inventory()` — `poc/runner/pipeline.py:547` |
| In | the inventory + `poc/ground_truth.csv` filtered by `--room` |
| Out | bias %, absolute error %, precision, recall |
| Printed | `stage 9  no ground truth for this room — not scored` |

**This is why you cannot yet judge output quality.** Nothing in `HomeObjects-3K` records
that a sofa is 1.1 m³. Without hand-measured rooms in `ground_truth.csv`, stage 9 has
nothing to compare against and the pipeline can only tell you what it *thinks* — never
whether it is right. This is gap **A1**, and it is the only blocker on a defensible
accuracy number.

---

## 2. Where every file lives

```
poc/
  data/input/          <- YOU PUT INPUT HERE
     demo_room.jpg           a single photo
     demo_bedroom/           a folder = several views of ONE room
  data/homeobjects3k/  <- the downloaded dataset (390 MB, gitignored)
  results/             <- EVERY RUN WRITES HERE
     <combo>__<input>__<UTC timestamp>.json
  detect_vocab.json    <- the 26 prompts, and which size classes each allows
  cube_table.json      <- 42 size classes with volumes  (PLACEHOLDER - not NX's real sheet)
  ground_truth.csv     <- DOES NOT EXIST YET. Stage 9 is silent until it does.
```

Output filenames are built from the configuration, so nothing overwrites anything:

```
grounding_dino__moge2__nomask__nocls__anchor__demo_room__20260907T035602Z.json
└─ detector ──┘ └depth┘ └mask─┘ └cls─┘ └anchor┘ └─input─┘ └── run time ──┘
```

You can always tell which models produced a result from its filename alone, and two
runs of the same config never collide.

---

## 3. Running on your own data

> **Prefer a pipeline folder.** Each pipeline in `poc/pipelines/<name>/` owns its own
> `data/input/` and `results/`, plus a CLI and a step-by-step notebook that read one
> `config.py`:
>
> ```bash
> mkdir -p poc/pipelines/grounding_dino__moge2__sonnet5/data/input/lounge
cp ~/photos/lounge/*.jpg poc/pipelines/grounding_dino__moge2__sonnet5/data/input/lounge/
> python -m poc.pipelines.grounding_dino__moge2__sonnet5.run          # one shot
> jupyter lab poc/pipelines/grounding_dino__moge2__sonnet5/notebook.ipynb   # stage by stage
> ```
>
> See `poc/pipelines/README.md`. The commands below are the general-purpose tool the
> pipeline folders are presets of — use them for one-off runs and ablations.


Nothing about the pipeline is HomeObjects-specific. That dataset is only used by
`run_dataset_eval`, because it happens to have boxes to score against.

**One photo:**
```bash
cp ~/Desktop/my_room.jpg poc/data/input/
python -m poc.runner.run_combination --input my_room.jpg \
    --detector grounding_dino --depth moge2 --classifier none
```

**One room, several photos** — the real product shape:
```bash
mkdir -p poc/data/input/smith_bedroom
cp ~/photos/bedroom/*.jpg poc/data/input/smith_bedroom/
python -m poc.runner.run_combination --input smith_bedroom \
    --detector grounding_dino --depth moge2 --classifier none
```

**A video walkthrough:**
```bash
cp ~/videos/walkthrough.mp4 poc/data/input/
python -m poc.runner.run_combination --input walkthrough.mp4 --max-frames 16 \
    --detector grounding_dino --depth moge2 --classifier none
```

**Any path at all**, no copying:
```bash
python -m poc.runner.run_combination --input /Users/me/anywhere/room.jpg ...
```

### One folder is one ROOM, not one dataset

Stages 7–8 deduplicate across frames on the assumption that every frame shows the
same room. Point it at 50 photos of 50 different rooms and it will merge them into a
single meaningless inventory. **One room per folder, one run per room.**

### Mind the warm-up when you loop

Grounding DINO's **first** inference in a process costs about **370 s**; every later
image costs 2.7 s. So a shell loop over rooms pays that 370 s *every time*:

```bash
# ~6 minutes per room, nearly all of it warm-up
for r in room1 room2 room3; do python -m poc.runner.run_combination --input $r ...; done
```

Ten rooms that way is over an hour of pure warm-up. Batch what you can into single
invocations, and for detector tuning use `run_dataset_eval --limit N`, which loads
the model once and reuses it.

---

## 4. Reading the output, and what the real run says about quality

From the run above:

```
(unnamed)          2.68 x 1.74 x 0.75 m  = 3.486 m3  -> sofa_sectional     (observed)
(unnamed)          2.22 x 0.76 x 0.64 m  = 1.086 m3  -> sofa_3_seat        (observed)
armchair           0.90 x 0.31 x 0.73 m  = 0.205 m3  -> armchair           (observed)
tv stand           1.93 x 1.07 x 0.74 m  = 1.522 m3  -> tv_stand           (observed)
rolled rug         2.01 x 0.99 x 0.53 m  = 1.057 m3  -> rug_rolled         (observed)
chest of drawers   1.93 x 1.02 x 0.73 m  = 1.438 m3  -> chest_5_drawer     (observed)
```

Read it critically, because the defects are visible without any ground truth:

- **A 1.74 m deep sofa does not exist.** Real sofas are 0.85–0.95 m deep. The box
  around it also contains floor and wall, and those pixels were measured as part of
  the object. Same story for the 1.07 m "tv stand" and the 1.02 m "chest of drawers".
- **Over-measured depth is the dominant error right now**, not classification. Every
  label above is plausible; the *dimensions* are what is wrong.
- **`armchair` at 0.31 m deep is the opposite failure** — too thin, because only the
  front face was visible.

The fix is already built and unused: **`--segmenter sam2`** turns each box into a
pixel mask, so only the object's own pixels are measured instead of everything in the
rectangle. Run the same image with and without it and compare `measured_raw_m3`:

```bash
python -m poc.runner.run_combination --input demo_room.jpg \
    --detector grounding_dino --depth moge2 --segmenter sam2 --classifier none
```

Two blocks in the JSON tell you how much to trust the rest, and both were added
because their absence hid real damage:

- **`label_quality`** — `{clean, narrowed, ambiguous, empty, unmatched}`. High
  `ambiguous` means lower `text_threshold`; high `empty` means raise it.
- **`depth_observability`** — `class_prior_pct`. At 0% every dimension was genuinely
  measured. Above ~60% Method B is mostly reading Method A's table and the two are no
  longer independent, so their agreement stops being evidence.

---

## 5. The knobs, and which way to turn them

| Flag | Default | Effect |
|---|---|---|
| `--detector` | `grounding_dino` | which model finds objects |
| `--depth` | `moge2` | which model estimates metres |
| `--segmenter sam2` | off | box to mask. **Biggest available accuracy win** |
| `--classifier none` | `claude_sonnet5` | `none` skips Method A: no API key, no cost |
| `--no-anchor` | off | disables the door correction — an ablation, to see what it buys |
| `--max-frames` | 16 | frames per room |
| `--dedup max\|median` | `max` | how to combine counts across frames |
| `--room <id>` | none | enables stage 9 scoring against `ground_truth.csv` |

Detector thresholds (`box_threshold`, `text_threshold`) are not yet CLI flags on
`run_combination`; `run_dataset_eval` exposes them as `--box-th` / `--txt-th`, which
is where threshold tuning belongs anyway since it can score the result.

---

## 6. What the current numbers do and do not mean

Your detection run said **precision 0.695, recall 0.222**. That is a genuine and
serious result: at `box_th=0.30` the detector misses roughly four objects in five.
`floor lamp` and `television` scored **0.000**, `dining table` 0.045, `chair` 0.090.
Only `sofa` (0.670) and `door` (0.571) work acceptably.

Two things follow, and it is worth keeping them apart:

1. **Recall is the priority, and thresholds are the first lever** — `box_th=0.30` is
   probably too high. `run_dataset_eval --compare-thresholds` sweeps it and shows the
   precision/recall trade directly. An object never detected can never be measured,
   so no amount of depth work recovers it.
2. **Detection recall is not accuracy.** Even at perfect recall, this dataset cannot
   tell you whether a volume is right, because it has no volume labels. The
   over-measured depths in section 4 would score perfectly on box IoU while producing
   a badly wrong quote.

So detection metrics tell you whether the pipeline can *see*. Only hand-measured rooms
tell you whether it can *count*. That is gap **A1**, and it stays open.

---

# Why the pipeline is shaped this way

> Merged in from the former `ARCHITECTURE.md`. Its pipeline overview, stage table,
> model list and file map all duplicated material above or in `MODELS.md`; what
> follows is the part that was only there — what the POC has to answer, the two
> competing methods, a worked example, and how a winner gets chosen.

## 1 · What the POC has to answer

> Point a camera at a room. Get back a list of what is in it and how many cubic metres it
> will take on a removal truck. **Then find out how wrong that answer is.**

That last sentence is the actual deliverable. Producing *a* number is easy; knowing how
much to trust it is the whole point.

---

## 2 · The two methods — the part that was unclear

There are two fundamentally different ways to get from a photograph to a volume. We do not
know which is better on our footage, so **we build both and race them once.**

### Method A · Recognise & Look Up

**Works like a surveyor with a clipboard.**

Look at the sofa. Decide *"that's a 3-seater."* Read the volume off a standard table:
`sofa_3_seat = 1.42 m³`. Write it down. Move on.

**It never measures anything.** The AI's only job is to put the right *name* on things.

### Method B · Measure & Compute

**Works like a surveyor with a tape measure.**

Work out the sofa is 2.08 m wide, 0.91 m deep, 0.86 m tall. Multiply. That is the volume.

**It never looks anything up.** The AI's only job is to *measure* accurately.

### Side by side

| | **Method A** | **Method B** |
|---|---|---|
| Mental model | surveyor with a clipboard | surveyor with a tape measure |
| The AI's job | name things correctly | measure things correctly |
| Core model | Claude Sonnet 5 (vision) | MoGe-2 (depth) |
| Needs the cube table? | **yes — completely dependent** | only for the final lookup |
| Needs to know scale? | **no** | **yes — and this is the hard part** |
| Fails when | the table is wrong, or the name is wrong | the scale is wrong, or the object is half hidden |
| Used by the industry? | **yes — all of them** | no product does this |
| Cost per room | ~$0.12 | ~$0 after setup |

### Why race them at all?

Because they fail for *different* reasons, and neither failure is obvious in advance.

- Method A inherits every error in the cube table. If NX's real 3-seater is 1.6 m³ and our
  table says 1.42 m³, Method A is 11% wrong **and cannot possibly detect it.**
- Method B inherits the depth model's scale error, published at **8.19%** — which becomes
  roughly **26% on volume**, because volume grows with the cube of length.

Every commercial product uses Method A. That is strong evidence, but it is *their* evidence,
not ours. One test settles it, then we delete the loser and stop paying for it.

---

## 7 · Worked example — one sofa, both routes

Illustrative numbers, to show the mechanics.

**Shared stages**

| Stage | What happens |
|---|---|
| 1 | Frame at `t=3.0s`, sharpness 142 — passes |
| 2 | Box `[412, 288, 901, 604]`, label `sofa`, score 0.71 |
| 3 | Pixels in that box sit 2.6–4.1 m from the camera |
| 4 | A door in the same frame measures **1.80 m**. Real doors are **1.981 m**. → `SCALE = 1.101` (everything was ~10% too small) |

**Then the paths split**

```
METHOD B · Measure & Compute            METHOD A · Recognise & Look Up
─────────────────────────────           ──────────────────────────────
points in box x 1.101                   send frame to Claude Sonnet 5
   height  (vertical axis)  0.86 m         |
   footprint (PCA)          2.08 x 0.91 m  v
   raw volume               1.72 m3      {"size_class": "sofa_3_seat",
   |                                       "count": 1,
   v                                       "confidence": 0.86}
nearest class among                        |
{2_seat, 3_seat, sectional}                v
   -> sofa_3_seat (dist 0.03)           cube_table -> 1.42 m3
   |
   v
cube_table -> 1.42 m3
```

Both land on 1.42 m³. **Now the interesting part — where they come apart:**

| Scenario | Method B | Method A |
|---|---|---|
| Sofa half behind a table | box is clipped, measures small, maps to `sofa_2_seat` → **1.0 m³, a 30% miss** | still reads "3-seater" from context → **correct** |
| No door in frame | no anchor, 8% scale error stands → **~26% volume error** | **unaffected** — it never used scale |
| Cube table wrong for NX | raw box volume is still independent evidence | **11% wrong and cannot detect it** |
| Unusual item not in the vocabulary | invisible | invisible (both fail — a vocabulary gap) |

This is the whole argument for building both. Stage 9 tells us which failure mode actually
dominates on real footage.

---

## 8 · How we decide the winner

Stage 9 prints two numbers per method. **Read them in this order:**

1. **`vol_bias_pct`** — is it consistently high or low? Consistent bias is one multiplier
   away from being fixed. Almost good news.
2. **`vol_abs_err_pct`** — how much does it vary job to job? Not fixable with a
   coefficient. The expensive problem.

**A method with large bias and small spread beats the reverse**, even if its raw error
looks worse. That is counter-intuitive and it is the single most important thing to
understand when reading the output.

Then run the notebook again with `use_scale_anchor = False` and diff. If the anchor buys
little, Method B's whole premise is weaker than expected.

---

## 9 · Deliberately not in the POC

Not oversights — tracked as Group C in `GAPS.md`:

carton and box counts · contents of closed cupboards · coverage enforcement ·
dismantling and fragile-packing rules · **browser upload UI** · authentication ·
persistence · back-office integration · GDPR controls · photo-archive security ·
self-pack vs full-pack · multi-room aggregation · confidence intervals and vehicle
planning · staff review console

The browser UI is deliberately last. It carries no technical risk and is about a day's work
once the pipeline produces a number worth showing.

---
