# Model Architectures — how each one actually thinks

> **FILE PURPOSE** — Plain-language intuition for every model in the POC, a diagram of
> what happens inside it, which file wraps it, and the exact command to run each
> combination.
>
> Diagrams are ASCII so they render in VS Code's built-in preview without an extension.
>
> Map: `ARCHITECTURE.md` · Licences: `MODELS.md` · Decisions: `GAPS.md`

---

## 0 · The only two questions any of these answer

Everything in the pipeline is one of two jobs:

```
  "WHAT is that, and WHERE?"          "HOW FAR AWAY is every pixel?"
   -> detectors                        -> depth models
   Grounding DINO, OWLv2, SAM 3,       MoGe-2, Depth Anything 3,
   RT-DETR, YOLO-World, YOLOE          UniDepthV2
```

A detector gives you a *shape on the image*. A depth model gives you *metres*. You need
both before you can say how big a sofa is.

---

# PART 1 · DETECTORS

## 1.1 · Grounding DINO — the baseline

**File:** `poc/models/detect_grounding_dino.py`
**Licence:** Apache 2.0 · **Speed:** slow · **Output:** boxes

### Intuition

Imagine reading a shopping list and scanning a room at the same time, going back and forth
— glance at the list, glance at the room, glance back — until each word on your list has
been matched to something you can see.

That "going back and forth" is the architecture. Image features and text features are
**fused repeatedly**, at several stages, rather than compared once at the end. That is why
it is accurate, and why it is slow.

```
   your words                          the photo
  "sofa. wardrobe. door."                 |
        |                                 |
        v                                 v
  +-------------+                  +--------------+
  | text encoder|                  | image encoder|
  |   (BERT)    |                  |   (Swin ViT) |
  +------+------+                  +-------+------+
         |                                 |
         |     +---------------------+     |
         +---->|  FEATURE  FUSION    |<----+     <-- text looks at image,
               |  cross-attention    |               image looks at text,
               |  (repeated)         |               back and forth
               +----------+----------+
                          |
                          v
               +---------------------+
               | language-guided     |   pick the image regions that
               | query selection     |   best match the words
               +----------+----------+
                          |
                          v
               +---------------------+
               |  decoder            |   refine each query into a box
               |  (DETR-style)       |   + which word it matched
               +----------+----------+
                          |
                          v
              [ sofa 0.71  box(412,288,901,604) ]
```

**Why it's the default:** highest zero-shot accuracy of the shippable detectors, Apache 2.0.

---

## 1.2 · OWLv2 — the long-tail specialist

**File:** `poc/models/detect_owlv2.py`
**Licence:** Apache 2.0 · **Speed:** slow · **Output:** boxes

### Intuition

Much simpler than Grounding DINO, and the simplicity is the point.

Cut the image into a grid of patches. **Every patch proposes one box and describes itself
as a vector.** Separately, turn your word into a vector. Then just compare: whichever
patches point in the same direction as your word are your matches.

No back-and-forth fusion. One pass, then a similarity check.

```
    the photo                              your word
        |                                  "wardrobe"
        v                                      |
  +--------------+                             v
  | ViT encoder  |                      +--------------+
  | image -> grid|                      | text encoder |
  |  of patches  |                      |   (CLIP)     |
  +------+-------+                      +------+-------+
         |                                     |
         v                                     |
  each patch gets:                             |
    * one box prediction                       |
    * one embedding  o---------+               |
                               |               |
                               v               v
                        +--------------------------+
                        |  dot product similarity  |
                        |  patch vector . word     |
                        +------------+-------------+
                                     |
                                     v
                        keep patches above threshold
```

**Why it's in the comparison:** trained with self-training on **over a billion**
pseudo-labelled examples, specifically to improve on **rare** categories. Our vocabulary is
full of rare things — dressing tables, rolled rugs, sideboards — so it may beat Grounding
DINO on exactly the items that matter to us, while losing on common-object benchmarks.

---

## 1.3 · SAM 3 — concept segmentation

**File:** `poc/models/detect_sam3.py`
**Licence:** Meta custom (commercial OK) · **Speed:** slow · **Output:** **masks**

### Intuition

You name a concept **once**, and it finds **every** instance of it and traces the
**outline** of each — not a rectangle, the actual silhouette.

```
   "chair"  (said once)
       |
       v
  +---------------+        +------------------+
  | concept       |------->| find ALL matching|
  | encoder       |        | instances        |
  +---------------+        +---------+--------+
       ^                             |
       |                    +--------+--------+--------+
  +----+----------+         v        v        v        v
  | image encoder |      chair 1  chair 2  chair 3  chair 4
  +---------------+         |        |        |        |
                            v        v        v        v
                       +--------------------------------+
                       |        mask decoder            |
                       | trace each object's outline    |
                       +--------------------------------+
```

### Why masks matter more than accuracy here

The single most important idea in the detector section.

```
   RECTANGLE around a sofa            MASK around a sofa
   +------------------------+         +------------------------+
   |::::: wall 6m back :::::|         |         wall           |
   |:::  +----------+   ::::|         |        +######+        |
   |:::  |  SOFA 3m |   ::::|         |        |#SOFA#|        |
   |:::  +----------+   ::::|         |        +######+        |
   |::::: floor :::::::::::::|        |         floor          |
   +------------------------+         +------------------------+
    measures 3m of "depth"             measures 0.9m of depth
    because the wall is inside         because only sofa pixels
    the box                            are counted
```

In testing a rectangle measured a synthetic object **45,000x too large** because the wall
behind it sat 3 metres further back. Every pixel inside the shape goes to the measurement
step, so background becomes part of the object.

**Setup note:** weights are gated. Accept the licence on the model page, then
`huggingface-cli login`.

---

## 1.4 · YOLO-World — open vocabulary at CNN speed

**File:** `poc/models/detect_yolo_world.py`
**Licence:** **AGPL-3.0 — CANNOT SHIP** · **Speed:** ~20x faster · **Output:** boxes

### Intuition

Start with a normal YOLO — a single-pass CNN that looks at the image once and predicts a
grid of boxes. Very fast, but its class list is welded in at training time.

YOLO-World's trick: **compile your words into the network before you run it.**

```
  SETUP (once, before any images)
      "sofa. wardrobe. door."
              |
              v
       +--------------+
       | CLIP text    |   turn each word into a vector
       | encoder      |
       +------+-------+
              |
              v
       +----------------------------+
       | bake the vectors into the  |   <-- "re-parameterisation"
       | network's own weights      |       the text encoder is now
       +------------+---------------+       DONE and gets thrown away
                    |
  ==================|=================================
  INFERENCE (per frame, at full CNN speed)
                    v
   photo  ->  +-------------+  ->  boxes + labels
              | YOLO CNN    |
              | (one pass)  |
              +-------------+
```

**That last step is why it is fast.** The expensive language model runs *once at setup*,
not per frame. At inference it is a plain CNN. Grounding DINO by contrast runs its text
fusion on every single frame.

**But:** AGPL-3.0 means shipping it would oblige NX to publish the entire source code of
the survey tool. Comparison only, and even that may need clearance. See `MODELS.md` §1b.

---

## 1.5 · YOLOE — the same trick, plus masks

**File:** `poc/models/detect_yoloe.py`
**Licence:** **AGPL-3.0 — CANNOT SHIP** · **Speed:** real-time · **Output:** **masks**

### Intuition

YOLO-World's speed, three ways to prompt it, and a segmentation head bolted on.

```
  THREE WAYS TO ASK
  ------------------------------------------------
  text     "wardrobe"                 -> find wardrobes
  visual   [crop of one wardrobe]     -> find things like THIS
  none     (built-in large vocab)     -> find everything you know
                    |
                    v
           +------------------+
           | YOLO CNN backbone|
           +--------+---------+
                    |
          +---------+---------+
          v                   v
      box head            MASK head      <-- the addition
      rectangles          outlines
```

On paper the best of everything: real-time, open vocabulary, *and* masks. Unshippable for
the same AGPL reason, so it exists purely to tell us what we are missing.

---

## 1.6 · RT-DETR — the licence-clean speed option

**File:** `poc/models/detect_rtdetr.py`
**Licence:** Apache 2.0 · **Speed:** real-time · **Output:** boxes · **Vocabulary: FIXED**

### Intuition

A transformer detector re-engineered to actually be fast. Classic DETR is elegant but slow
because attention runs over every scale at once. RT-DETR splits that: heavy attention only
on the smallest feature map, then fuse scales cheaply.

```
   photo
     |
     v
  +----------+
  | backbone |   multi-scale features
  +----+-----+
       |
       +--- small map ---> [ self-attention ]  <-- expensive, but tiny input
       |                          |
       +--- large maps -----------+
                    |
                    v
          +--------------------+
          | cheap cross-scale  |
          | fusion (CNN)       |
          +---------+----------+
                    |
                    v
          +--------------------+
          | IoU-aware query    |   pick promising regions
          | selection          |   instead of brute force
          +---------+----------+
                    |
                    v
          +--------------------+
          | decoder -> boxes   |   no NMS needed
          +--------------------+
```

**The trade.** It only knows the classes it was trained on — Objects365 (365 classes) plus
COCO. Far more household objects than COCO's 80, but you still cannot ask for something
new. The adapter records which of our prompts it **cannot even attempt**, so the comparison
stays honest.

**Why run it:** it answers "what does open vocabulary actually buy us?" If a fixed
365-class model covers 90% of a survey, speed and licence clarity may be worth more than
flexibility.

---

# PART 2 · DEPTH MODELS

## 2.1 · MoGe-2 — the default

**File:** `poc/models/depth_moge2.py`
**Licence:** MIT · **Error:** 8.19% metric scale

### Intuition — the most important diagram in this document

MoGe-2 splits the problem in two, and the split explains the entire project.

> **One part works out the SHAPE of the scene, very precisely.**
> **Another part looks at the whole image and guesses HOW BIG it is.**
> **Multiply them together.**

```
    photo
      |
      v
 +-------------+
 | DINOv2 ViT  |
 |   encoder   |
 +------+------+
        |
        +-----------------------------+
        |                             |
        v                             v
 +---------------+          +-----------------------+
 | relative      |          | metric scale head     |
 | geometry head |          | (MLP off the ONE      |
 | (conv decoder)|          |  global CLS token)    |
 +-------+-------+          +----------+------------+
         |                             |
         v                             v
   point map with            +-------------------+
   correct SHAPE but         |  ONE  NUMBER      |
   unknown size              |  the scene scale  |
         |                   +---------+---------+
         |                             |
         +----------> x <--------------+
                      |
                      v
            metric point map (metres)
```

### Why this is the weak link

Look at what carries the scale: **a single scalar, predicted from one global token.**

The shape is estimated from thousands of pixels and is very good. The *size* is one guess
from one summary vector — and that one number multiplies every measurement in the room.

Which is exactly why:

- scale error is **systematic**, not random — it cannot average out across objects
- an 8% scale error becomes **~26% volume error**, because volume goes as length cubed
- **the door anchor works so well** — you replace one guessed scalar with one measured one

Stage 4 of the pipeline is not a refinement. It is a direct substitution for the single
weakest number in the whole system.

---

## 2.2 · Depth Anything 3 — the multi-view option

**File:** `poc/models/depth_anything3.py`
**Licence:** Apache 2.0 *(metric variant only — see below)*

### Intuition

One transformer that looks at **several photos at once**. It alternates between "study this
photo on its own" and "compare across photos", just by reshuffling which tokens attend to
which.

```
  frame1  frame2  frame3   (a slow pan gives you these free)
    |       |       |
    +-------+-------+
            |
            v
   +---------------------+
   |  block 1: WITHIN    |   each frame studies itself
   |  block 2: ACROSS    |   frames compare with each other
   |  block 3: WITHIN    |
   |  block 4: ACROSS    |   ... alternating
   +----------+----------+
              |
       +------+------+
       v             v
   depth map     ray map
```

**Why it matters:** a guided video pan gives genuine multi-view geometry for free. Two views
of the same sofa constrain its size in a way one photo never can. If DA3 beats MoGe-2 on
our footage, the "slow pan" capture instruction pays for itself twice.

> **LICENCE TRAP — the variant matters, not the project.**
> `DA3METRIC-LARGE`, `DA3MONO-LARGE`, `DA3-BASE`, `DA3-SMALL` = **Apache 2.0**, fine.
> `DA3-LARGE`, `DA3-GIANT`, `DA3NESTED` = **CC BY-NC**, cannot ship.
> The adapter defaults to the Apache *metric* one deliberately.

---

## 2.3 · UniDepthV2 — the research ceiling

**File:** `poc/models/depth_unidepth.py`
**Licence:** **CC BY-NC 4.0 — CANNOT SHIP** · Run via `poc/rnd/`

### Intuition

Most depth models guess an (x, y, z) point per pixel directly. UniDepth splits that into
two easier questions:

1. **Which direction does this pixel point?** (a ray — angle up/down, angle left/right)
2. **How far along that ray is the surface?** (one distance)

Separating direction from distance is why it generalises across cameras: the camera affects
the directions, not the distances.

```
    photo
      |
      v
 +-------------+
 | ViT encoder |
 +------+------+
        |
        +----------------------------+
        v                            v
 +----------------+        +--------------------+
 | camera module  |        | depth module       |
 | which way does |        | how far along that |
 | each pixel     |        | ray?               |
 | point?         |        |                    |
 +-------+--------+        +---------+----------+
         |                           |
      azimuth, elevation          log-depth
         |                           |
         +------------+--------------+
                      v
        pseudo-spherical output (theta, phi, log z)
        -- camera geometry and depth kept SEPARATE
```

**In the POC purely to answer one question:** how much accuracy are we giving up by only
using models we can ship? Run it from `poc/rnd/`, never the normal sweep.

---

# PART 3 · SUPPORTING MODELS

## 3.1 · SAM 2 — turns any box into a mask

**File:** `poc/models/segment_sam2.py` · **Licence:** Apache 2.0

Give it a rectangle; it grows the region that actually belongs to the object inside.

```
  photo + box from ANY detector
          |
          v
   +---------------+
   | image encoder |
   +-------+-------+
           |          +----------------+
           +--------->| mask decoder   |---> outline
                      +----------------+
                            ^
                      box as a prompt
```

**Why it earns its place:** it gives Grounding DINO, OWLv2 and RT-DETR the same mask
advantage SAM 3 and YOLOE have built in. One extra pass, and the background-pixel
over-measurement goes away. Run 8 of the sweep measures whether that trade is worth it.

## 3.2 · Claude Sonnet 5 — Method A

**File:** `poc/models/classify_claude.py` · commercial API

```
  frame ---> vision encoder ---> tokens ---> LLM ---> forced tool call
                                                            |
                                     +----------------------+
                                     v
                     { "size_class": "sofa_3_seat",   <-- enum = our cube table,
                       "count": 1,                        so an invalid class is
                       "confidence": 0.86 }                impossible by construction
```

It **names**, it does not count. Measured VLM counting accuracy is ~0.53 and it
under-counts, producing confident low inventories and undersized trucks.

---

# PART 4 · THE PIPELINE SHAPES

The 14 combinations reduce to **six distinct data-flow shapes**. Combinations sharing a
shape share a diagram — drawing 14 near-identical pictures would hide the differences
rather than show them.

### Shape A — baseline: box detector + depth

Runs **1, 3, 5, 9, 11, 12** · *(also 6, 10 — not shippable)*

```
 frames --+--> [detector] --boxes--+--> [scale anchor] --factor--+
          |                        |                             |
          +--> [depth]  --points---+-----------------------------+--> [measure]
          |                                                            |
          +--> [Claude] --names----------------------+                 |
                                                     v                 v
                                            [dedup] -> [cube table] -> [score]
```

Boxes only, so measurements include background pixels. The reference everything else is
judged against.

### Shape B — native mask detector

Runs **4, 7** *(7 not shippable)*

```
 frames --+--> [SAM 3 / YOLOE] --boxes + MASKS--+--> [scale anchor] --+
          |                                     |                     |
          +--> [depth] -------------------------+---------------------+--> [measure]
                                                                           ^
                                                     masks restrict measurement
                                                     to the object's own pixels
```

The only change from Shape A: outlines instead of rectangles, which removes the
background-pixel over-measurement at source.

### Shape C — box detector + mask refinement

Runs **8** and **14**

```
 frames --> [detector] --boxes--> [SAM 2] --MASKS--> [scale anchor] --> [measure]
                                     ^
                          one extra model pass per frame
```

Same end state as Shape B, reached by bolting SAM 2 onto a box detector. **Run 8 vs run 4**
answers: is a bolted-on mask as good as a native one, and is it worth the extra pass?

### Shape D — anchor off (the key experiment)

Run **2**

```
 frames --+--> [detector] --boxes--+
          |                        |
          +--> [depth] --points----+--> [measure]        <-- NO scale correction
                                          ^
                              raw model scale, ~8% error
                              -> ~26% on volume
```

Identical to Shape A except the door anchor is bypassed. **The gap between run 1 and run 2
is the entire value of the anchor.**

### Shape E — no classifier, geometry alone

Run **13**

```
 frames --+--> [detector] --boxes--+--> [scale anchor] --> [measure] --> [cube table]
          |                        |
          +--> [depth] ------------+
                                            (Claude never called)
```

What pure geometry achieves with no language model at all. Also free — no API cost.

### Shape F — closed vocabulary

Run **5**

```
  our 26 prompts
        |
        v
  +--------------------+
  | alias map          |  17 prompts map to an RT-DETR class
  | 9 have NO class    |  9 are UNREACHABLE and recorded as such
  +---------+----------+
            v
      [RT-DETR] --> boxes --> (continues as Shape A)
```

Structurally Shape A, but some prompts can never be found. The adapter reports them so the
comparison is not silently unfair to a model that was never allowed to try.

---

# PART 5 · RUNNING EACH COMBINATION

All commands assume footage in `poc/data/input/` and `poc/ground_truth.csv` filled in.

```bash
# --- 1 · baseline (Shape A) ----------------------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5

# --- 2 · anchor off · THE KEY EXPERIMENT (Shape D) ------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 --no-anchor

# --- 3 · OWLv2, long-tail detector (Shape A) ------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector owlv2 --depth moge2 --classifier claude_sonnet5

# --- 4 · SAM 3, native masks (Shape B) ------------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector sam3 --depth moge2 --classifier claude_sonnet5

# --- 5 · RT-DETR, closed vocab (Shape F) ----------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector rtdetr --depth moge2 --classifier claude_sonnet5

# --- 8 · SAM 2 mask refinement (Shape C) ----------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --segmenter sam2 --classifier claude_sonnet5

# --- 9 · Depth Anything 3 (Shape A) ---------------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth da3_metric --classifier claude_sonnet5

# --- 11 / 12 · classifier size (Shape A) ----------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_opus5
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_haiku45

# --- 13 · geometry only, no LLM (Shape E) ---------------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier none

# --- 14 · best guess stack (Shape C) · run LAST ---------------------------
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector sam3 --depth moge2 --segmenter sam2 --classifier claude_sonnet5
```

### All of them at once

```bash
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --dry-run
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --skip-unavailable
python -m poc.runner.compare   --room BED01
```

### Runs 6, 7, 10 — the ones that cannot ship

Deliberately not in the list above. They live in quarantine:

```bash
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 --dry-run
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 \
    --i-have-legal-clearance
```

Results go to `poc/rnd/results/`, tagged `shippable: false`, and `compare.py` never reads
them. See `poc/rnd/README.md`.

---

## Reading the output

Each run prints its stages and writes one JSON to `poc/results/`. Three numbers matter:

| Field | Means | How to read it |
|-------|-------|----------------|
| `vol_bias_pct` | consistently high or low? | **read this first** — one multiplier from fixed |
| `vol_abs_err_pct` | how much it varies job to job | the expensive problem |
| `class_prior_pct` | how often depth came from the cube table, not the camera | **high = Method B is not measuring independently** (gap B8) |

---

## Which file does what

| Concern | File |
|---------|------|
| The contracts that make models swappable | `poc/models/base.py` |
| Catalogue, availability, **licence gate** | `poc/models/registry.py` |
| Detectors | `poc/models/detect_*.py` |
| Depth estimators | `poc/models/depth_*.py` |
| Box to mask | `poc/models/segment_sam2.py` |
| Method A classifier | `poc/models/classify_claude.py` |
| Stage maths, model-agnostic | `poc/runner/pipeline.py` |
| Run one combination | `poc/runner/run_combination.py` |
| Run the curated 14 | `poc/runner/run_sweep.py` |
| Rank the results | `poc/runner/compare.py` |
| Non-shippable models, quarantined | `poc/rnd/run_research_ceiling.py` |
