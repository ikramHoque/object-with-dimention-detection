# Models & Combinations

> **FILE PURPOSE** — Every model the POC can run, what each one is for, what it costs in
> licence terms, and how to race them against each other.
>
> Map: `PIPELINE.md` · Open decisions: `GAPS.md` · Build state: `STATUS.md`

---

## 1 · What "YOLO" actually means

"YOLO" is not one model. It is a family, and the difference between its members is the
whole answer to your question.

### YOLO, classic (v5, v8, v11, v12…)

Extremely fast object detector. Trained on a dataset called **COCO**, which contains
**80 fixed classes**. The model has exactly 80 output slots — one per class — and that is
permanently what it can say.

COCO's 80 classes include `chair`, `couch`, `bed`, `dining table`, `tv`, `refrigerator`,
`microwave`. They do **not** include:

> wardrobe · chest of drawers · sideboard · dressing table · bookcase · mattress ·
> washing machine · dishwasher · cardboard box · rolled rug · **door**

You cannot ask classic YOLO for a wardrobe. There is no slot for it. To add one you must
collect and label thousands of wardrobe photographs and retrain the model — weeks of work
per class. For a removal survey, where the missing classes are half the volume of the
shipment, that is fatal. **This was my original reason for not using it, and that part
stands.**

### YOLO-World (2024)

Someone bolted a **text encoder** onto YOLO. Now you pass it *words* and it finds those
things — no retraining. Ask for "wardrobe" and it looks for a wardrobe.

This is called **open-vocabulary** detection, and it is the same trick Grounding DINO uses.
The difference is speed: YOLO-World runs at **52 frames per second** where Grounding DINO
manages a few, so roughly **20× faster at a fifth of the size**, with competitive accuracy.

**This solves my original objection completely.** Which is why leaving it out was
under-justified, and why it is now in the comparison.

### YOLOE (2025-26)

Same open-vocabulary idea, newer, and it adds one thing that matters a lot here: it outputs
**masks** rather than rectangles.

A rectangle around a sofa also contains the wall behind it and the floor in front. Our
measurement step takes every 3D point inside the shape and measures how far it spreads — so
with a rectangle, the wall gets measured as part of the sofa. In testing, a rectangle
measured a synthetic object **45,000× too large** because the wall behind it was 3 metres
further back. A mask traces the object's actual outline and that error disappears.

### Where this leaves us

| | Open vocabulary? | Masks? | Speed | Can NX ship it? |
|---|---|---|---|---|
| YOLO classic | **no** — 80 fixed classes | no | fastest | yes, but useless to us |
| YOLO-World | **yes** | no | very fast | **no — see §1b** |
| YOLOE | **yes** | **yes** | very fast | **no — see §1b** |
| Grounding DINO | yes | no | slow | **yes** |
| SAM 3 | yes | **yes** | slow | **yes** |
| RT-DETR | no — fixed, but 365 classes | no | very fast | **yes** |

---

## 1b · The licence problem, from scratch

This is the part that decides whether we can use YOLO-World or YOLOE at all, so it is worth
being precise about.

### Who is Ultralytics?

A **company**. They maintain the `ultralytics` Python package — `pip install ultralytics` —
which is how essentially everyone runs modern YOLO, including YOLO-World and YOLOE. The
research papers come from elsewhere (Tencent, academia); Ultralytics packages the code you
would actually import.

So "using YOLO-World" in practice means "using Ultralytics' software."

### What a software licence actually controls

Open-source does not mean unconditional. Every model and library comes with a licence, and
there are three kinds that matter to us:

**Type 1 — Permissive: MIT, Apache 2.0**
> *"Use this however you like — in a paid product, closed-source, anything. Just keep our
> copyright notice somewhere."*

No obligations that affect NX. This covers **MoGe-2** (MIT), **Grounding DINO**, **OWLv2**,
**RT-DETR** and **SAM 2** (all Apache 2.0). These are safe and need no further thought.

**Type 2 — Copyleft: AGPL-3.0** ← *this is Ultralytics*
> *"Use this freely, but if you distribute your software or run it as a network service,
> you must publish YOUR ENTIRE SOURCE CODE under this same licence."*

Read that again with NX in mind. The survey tool is a **web application** — a network
service. Under AGPL, NX would be obliged to publish the complete source code of it,
publicly, for anyone to take: the pipeline, the cube-sheet logic, the pricing rules, the
lot. For a commercial logistics company that is simply not an option.

AGPL is deliberately stricter than the more familiar GPL precisely to close the
"we only expose it over the network" loophole. Hosting it as an API is not an escape.

**The paid escape.** Ultralytics sell an **Enterprise Licence** which removes the
publish-your-source obligation. Price is not published; it is negotiated per company.

**The extra sting.** Ultralytics' own published position is that the Enterprise Licence is
required for *any* use of their models — **explicitly including internal R&D, whether or
not it is commercial**. On a strict reading, even running our POC comparison needs it.

**Type 3 — Non-commercial: CC BY-NC 4.0**
> *"Research and personal use only. No commercial use at all."*

No paid escape is offered. This covers **UniDepthV2** and the large **Depth Anything 3**
variants. They can be used to measure what we are giving up, and never shipped.

### The three licences side by side

| Licence | Can NX ship a closed-source product? | Obligation | Our models |
|---------|--------------------------------------|------------|------------|
| **MIT / Apache 2.0** | **Yes** | keep the copyright notice | MoGe-2, Grounding DINO, OWLv2, RT-DETR, SAM 2 |
| **AGPL-3.0** | **No**, unless you pay | publish your entire source code | YOLO-World, YOLOE (via Ultralytics) |
| **CC BY-NC 4.0** | **No**, and no price exists | non-commercial use only | UniDepthV2, DA3-Large/Giant |

### What we do about it

The registry enforces this rather than documenting it. `yolo_world`, `yoloe` and
`unidepth2` are marked `commercial_ok=False`, and:

```
$ python -m poc.runner.run_combination --detector yolo_world --input bedroom.mp4

REFUSED — this combination contains models that cannot ship:
  yolo_world: AGPL-3.0 or paid Ultralytics Enterprise
      AGPL-3.0. Ultralytics require an Enterprise Licence even for internal
      R&D unless the whole project is open-sourced. Comparison use only.

Pass --allow-noncommercial to run it anyway for comparison purposes.
The result will be tagged shippable=false.
```

Results from such a run carry `shippable: false` permanently, and `compare.py` prints them
under a **CANNOT SHIP** heading so nobody builds a plan on them by accident.

**Recommendation.** Default to Grounding DINO, OWLv2 or SAM 3 — all shippable, all open
vocabulary. If speed becomes the binding constraint, use **RT-DETR** (Apache 2.0,
real-time); the cost is a fixed vocabulary, and the adapter records which of our prompts it
cannot even attempt so the comparison stays honest.

Run YOLO-World and YOLOE **only as benchmarks**, and only after NX's legal team has given a
view — because Ultralytics' stated position covers internal R&D, the comparison run itself
may need clearance. Buy the Enterprise Licence only if the sweep proves the speed advantage
is large *and* NX's survey volume makes it worth paying for. Tracked as gap **B9**.

---

## 2 · The full catalogue

Run `python -m poc.models.registry` to print this live, with installed/not-installed status.

### Detectors

| Key | Model | Licence | Ship? | Vocab | Masks | Why it's here |
|-----|-------|---------|-------|-------|-------|---------------|
| `grounding_dino` | Grounding DINO base | Apache-2.0 | **yes** | open | no | **The baseline.** Zero-shot accuracy leader, 52.5% AP on COCO with no COCO training |
| `owlv2` | OWLv2 base ensemble | Apache-2.0 | **yes** | open | no | Trained for **rare and long-tail** categories — which is exactly our vocabulary |
| `sam3` | SAM 3 | Meta licence | **yes** | open | **yes** | Native masks, 270k concepts. Most promising option. Weights gated on HF |
| `rtdetr` | RT-DETR R50 (O365+COCO) | Apache-2.0 | **yes** | **fixed** | no | Real-time *and* licence-clean. Measures what open vocabulary is actually worth |
| `yolo_world` | YOLO-World v2-L | **AGPL-3.0** | **NO** | open | no | ~20× faster. Comparison only |
| `yoloe` | YOLOE-v8L-seg | **AGPL-3.0** | **NO** | open | **yes** | Real-time + open vocab + masks. Best on paper, unshippable |

### Depth

| Key | Model | Licence | Ship? | Why it's here |
|-----|-------|---------|-------|---------------|
| `moge2` | MoGe-2 ViT-L | **MIT** | **yes** | **The default.** Best shippable metric accuracy: 8.19% point-map error |
| `da3_metric` | Depth Anything 3 Metric-Large | Apache-2.0 | **yes** | Handles multi-view, which a video pan gives us for free |
| `unidepth2` | UniDepthV2 ViT-L | **CC BY-NC** | **NO** | Best indoor numbers in the field. Here purely to measure what the licence costs us |

> **DA3 licence trap:** the variant matters, not the project. `DA3METRIC-LARGE`,
> `DA3MONO-LARGE`, `DA3-BASE` and `DA3-SMALL` are Apache 2.0. `DA3-LARGE`, `DA3-GIANT`
> and the `NESTED` variants are CC BY-NC. Do not swap to DA3-LARGE for "better numbers."

### Segmenter and classifiers

| Key | Model | Licence | Why it's here |
|-----|-------|---------|---------------|
| `sam2` | SAM 2.1 Hiera-Large | Apache-2.0 | Upgrades any box detector to masks. Cheap fix for the background-pixel problem |
| `claude_sonnet5` | Claude Sonnet 5 | commercial API | **Method A default.** ~$0.02/frame |
| `claude_opus5` | Claude Opus 5 | commercial API | Is 2.5× the cost worth it on hard rooms? |
| `claude_haiku45` | Claude Haiku 4.5 | commercial API | Is a fifth of the cost good enough? |

**3 of 13 models cannot ship.** The registry is the gate, not a comment.

---

## 3 · The three axes

Each axis answers a different question. The sweep changes **one at a time**.

| Axis | Options | The question |
|------|---------|--------------|
| **Detector** | 6 | Which finds our furniture best? Do masks beat boxes? Is open vocabulary worth the speed? |
| **Depth** | 3 | Does anything beat MoGe-2 — and what does the licence cost us in accuracy? |
| **Masks** | on / off | Does removing background pixels fix Method B's over-measurement? |
| **Classifier** | 4 | How much model do we need to pick a size class? |
| **Scale anchor** | on / off | **THE KEY EXPERIMENT.** What is the door worth? |

### Why not every combination

The full grid is **6 × 3 × 2 × 4 × 2 = 288 runs**, roughly **19 hours** on an M1, and most
cells answer no question anyone asked.

`combinations.json` instead defines **14 one-factor-at-a-time runs**: start from the
baseline and change exactly one thing, so a difference in the result has exactly one
possible cause. 14 runs, 14 answers. If two factors turn out to interact, add a targeted
pair afterwards — don't grid pre-emptively.

---

## 3b · The 14 combinations, in full

Generated from `combinations.json`. Every run is the baseline with **exactly one thing
changed**, so any difference in the result has exactly one possible cause.

| # | Run | Changed from baseline | Detector | Depth | Mask | Classifier | Anchor | Ship? |
|---|-----|----------------------|----------|-------|------|------------|--------|-------|
| 1 | `baseline` | — | grounding_dino | moge2 | — | claude_sonnet5 | on | yes |
| 2 | `anchor_off` | scale anchor | grounding_dino | moge2 | — | claude_sonnet5 | **OFF** | yes |
| 3 | `det_owlv2` | detector | owlv2 | moge2 | — | claude_sonnet5 | on | yes |
| 4 | `det_sam3` | detector | sam3 | moge2 | — | claude_sonnet5 | on | yes |
| 5 | `det_rtdetr` | detector | rtdetr | moge2 | — | claude_sonnet5 | on | yes |
| 6 | `det_yolo_world` | detector | yolo_world | moge2 | — | claude_sonnet5 | on | **NO** |
| 7 | `det_yoloe` | detector | yoloe | moge2 | — | claude_sonnet5 | on | **NO** |
| 8 | `mask_sam2` | segmenter | grounding_dino | moge2 | sam2 | claude_sonnet5 | on | yes |
| 9 | `depth_da3` | depth | grounding_dino | da3_metric | — | claude_sonnet5 | on | yes |
| 10 | `depth_unidepth_ceiling` | depth | grounding_dino | unidepth2 | — | claude_sonnet5 | on | **NO** |
| 11 | `cls_opus5` | classifier | grounding_dino | moge2 | — | claude_opus5 | on | yes |
| 12 | `cls_haiku45` | classifier | grounding_dino | moge2 | — | claude_haiku45 | on | yes |
| 13 | `method_b_only` | classifier | grounding_dino | moge2 | — | none | on | yes |
| 14 | `best_guess_stack` | combined | sam3 | moge2 | sam2 | claude_sonnet5 | on | yes |

### What each run is asking

**1. `baseline`** — What does the licence-clean default configuration achieve?
**2. `anchor_off`** — THE KEY EXPERIMENT. How much does the door anchor actually buy?
**3. `det_owlv2`** — Does OWLv2's long-tail training beat Grounding DINO on unusual furniture?
**4. `det_sam3`** — Do SAM 3's native masks improve Method B over boxes alone?
**5. `det_rtdetr`** — How much do we lose by giving up open vocabulary for speed and a clean licence?
**6. `det_yolo_world`** — Is YOLO-World's 20x speed advantage free, or does accuracy drop? AGPL - comparison only.
**7. `det_yoloe`** — Does YOLOE deliver real-time open vocabulary AND masks? AGPL - comparison only.
**8. `mask_sam2`** — Does bolting SAM 2 masks onto a box detector match a native mask model?
**9. `depth_da3`** — Does Depth Anything 3's metric variant beat MoGe-2 on our footage?
**10. `depth_unidepth_ceiling`** — CEILING CHECK. How much accuracy does staying licence-clean cost us? CC BY-NC - cannot ship.
**11. `cls_opus5`** — Is Opus 5 worth 2.5x the cost for size-class accuracy?
**12. `cls_haiku45`** — Is Haiku 4.5 good enough at a fifth of the cost?
**13. `method_b_only`** — How does pure geometry do with no language model at all?
**14. `best_guess_stack`** — Everything we expect to be best, together. Run this LAST, after the single-factor runs point at it.

### Reading the table

- **Run 1** is the reference. Everything else is compared against it.
- **Run 2** is the one that matters most: identical to the baseline apart from the scale
  anchor being switched off. The gap between rows 1 and 2 is the value of the anchor, and
  `compare.py` reports it as a matched pair for exactly this reason.
- **Runs 6, 7 and 10** are marked **NO** under Ship — they contain AGPL or non-commercial
  models. They exist to tell us what we are giving up by staying licence-clean, and they
  can never become the product. The runner refuses them without `--allow-noncommercial`.
- **Run 13** removes the language model entirely, so it measures what pure geometry
  achieves on its own.
- **Run 14** stacks everything we *expect* to be best. Run it **last** — once the
  single-factor runs have shown whether those expectations were right.

### Why not all 288

The full grid is 6 detectors × 3 depth models × 2 mask settings × 4 classifier settings ×
2 anchor settings = **288 runs**, about **19 hours** on an M1. And it would be mostly
wasted: if you change four things at once and the number moves, you have learned nothing
about which of the four did it.

Changing one factor at a time is how you attribute a result to a cause. 14 runs, 14
answers. If two factors turn out to interact — say masks only help with a particular depth
model — add that specific pair afterwards.

---

## 4 · Running it

```bash
python -m poc.models.registry                      # what exists, what's installed

# one combination
python -m poc.runner.run_combination \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 \
    --input bedroom.mp4 --room BED01

# THE KEY EXPERIMENT — same thing, anchor off
python -m poc.runner.run_combination \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 \
    --input bedroom.mp4 --room BED01 --no-anchor

# the curated sweep
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --dry-run
python -m poc.runner.run_sweep --input bedroom.mp4 --room BED01 --skip-unavailable

# rank everything
python -m poc.runner.compare --room BED01
```

Each run writes one JSON to `poc/results/`. `compare.py` loads them all and ranks by
**bias**, showing spread alongside — because a method that is consistently 15% low is one
multiplier from fixed, and one that is randomly ±15% is not.

`compare.py` pairs anchor-on against anchor-off runs **with identical config**. Comparing
the best anchored run against the best un-anchored one would mix in the detector choice
and credit the anchor with someone else's gain.

---

## 5 · Two defects the tests found

Both were caught by testing the maths on synthetic geometry before any real footage, and
both were real.

### The door measurement was biased ~4% high

`extent()` uses 2nd–98th percentiles to reject depth outliers at object edges. The door
measurement inherited that, but a door box is tight and has few outliers — so the clipping
was shaving ~4% off the true height, **inflating the scale factor by 4% and the volume by
~13%**. A systematic bias, in the exact term the anchor exists to remove.

Fixed by using 0.5–99.5 percentiles for the door specifically. Verified: scale error
dropped from **4.2% to 1.0%**, volume bias from ~13% to 3.1%.

### Cameras only see the front of things

More serious, and not predicted. A camera never sees the back of a wardrobe. The depth
model returns only the **visible surface**, so a flat-fronted object's point cloud is a
thin sheet with no measurable depth — and PCA correctly reports a depth of ~0, collapsing
the volume to near zero.

This is a hard limit of single-view geometry, and it hits most furniture: wardrobes,
chests, bookcases, fridges, anything against a wall. **Method B systematically
under-measures depth on exactly the items carrying the most volume.**

`resolve_dims()` now detects it (second principal axis under 12% of the variance), matches
on the two dimensions that *were* observed, and adopts that class's typical depth. Verified:
a wardrobe-shaped flat surface resolves to `wardrobe_double` at 0.60 m rather than 0.00 m.

**But watch `class_prior_pct` in every result.** When depth comes from the cube table,
Method B is using Method A's data and the two methods are no longer independent. The runner
warns above 60%. If it *is* high on real footage, the honest conclusion is that pure
geometric measurement is not viable from single-view capture — which is itself a headline
POC finding. Tracked as gap **B8**.

---

## 6 · File map

| File | Purpose |
|------|---------|
| `models/base.py` | The contracts. What makes models swappable |
| `models/registry.py` | key → adapter, plus licence and availability. **The licence gate** |
| `models/detect_*.py` | Six detectors, one file each |
| `models/depth_*.py` | Three depth estimators |
| `models/segment_sam2.py` | Box → mask refinement |
| `models/classify_claude.py` | Method A, three model sizes |
| `runner/pipeline.py` | Stage functions with every model choice removed |
| `runner/run_combination.py` | Run one combination → one result JSON |
| `runner/run_sweep.py` | Run the curated set |
| `runner/compare.py` | Rank everything |
| `combinations.json` | The 14 presets, each with the question it answers |
| `results/` | One JSON per run. Gitignored — these can be large |

---

# How each model actually thinks

> Merged in from the former `ARCHITECTURES.md`, which sat beside `PIPELINE.md`
> under a nearly identical name and was impossible to tell apart. Model internals
> belong with the model catalogue; the pipeline lives in `PIPELINE.md`.

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
