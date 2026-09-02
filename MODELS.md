# Models & Combinations

> **FILE PURPOSE** — Every model the POC can run, what each one is for, what it costs in
> licence terms, and how to race them against each other.
>
> Map: `ARCHITECTURE.md` · Open decisions: `GAPS.md` · Build state: `STATUS.md`

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
