# Learning the pipeline — one model at a time

> **FILE PURPOSE** — Intuition and **parameter tuning** for every model we use. Not theory:
> the aim is that you can look at a bad output, know which knob caused it, and predict what
> happens when you turn it.
>
> Each lesson has the same shape: **what it does → the knobs → how to diagnose a bad result.**

| # | Lesson | Status |
|---|--------|--------|
| 0 | Vocabulary — the terms that keep coming up | ✅ |
| 1 | Grounding DINO — detection | ✅ |
| 2 | OWLv2 — the simpler contrast | pending |
| 3 | SAM 2 / SAM 3 — masks | pending |
| 4 | YOLO-World / YOLOE | pending |
| 5 | RT-DETR | pending |
| 6 | MoGe-2 — depth | pending |
| 7 | Depth Anything 3 / UniDepthV2 | pending |
| 8 | Claude vision — Method A | pending |

---

# Lesson 0 · Vocabulary

The five terms that appear in every model's settings.

## Confidence / score

Every detection comes with a number 0–1: *"how sure am I there's an object here."* Not a
probability of being right — just the model's internal enthusiasm. Calibration varies by
model, which is why the same threshold behaves differently on Grounding DINO than on OWLv2.

## IoU — Intersection over Union

How much two boxes overlap.

```
   IoU = shared area / combined area

   +--------+                +--------+           +--------+
   |   A    |                |  A  +--+--+        |  A/B   |
   |    +---+---+            |     |  |  |        |        |
   +----+---+   |            +-----+--+  |        +--------+
        |   B   |                  |  B  |
        +-------+                  +-----+

   IoU ~ 0.15                  IoU ~ 0.45          IoU = 1.0
   barely touching             substantial         identical
```

Everything about duplicate removal is expressed in IoU.

## NMS — Non-Maximum Suppression

**The problem it solves.** Detectors fire more than once on the same object. Point one at a
sofa and you may get five overlapping boxes, all saying "sofa".

**What it does.** Sort every box by confidence. Take the most confident. Delete every other
box *of the same class* that overlaps it more than the IoU threshold. Move to the next
survivor. Repeat.

```
  BEFORE NMS                          AFTER NMS (iou_thresh = 0.65)
  +-----------+                       +-----------+
  | sofa 0.91 |                       | sofa 0.91 |   <- kept, highest score
  +-----------+                       +-----------+
   +-----------+                       (deleted, IoU 0.88 with the winner)
   | sofa 0.87 |
   +-----------+
                +-----------+                        +-----------+
                | sofa 0.83 |                        | sofa 0.83 |  <- kept, far away
                +-----------+                        +-----------+
```

**Where it hits us:** counts go straight into volume. Get NMS wrong and the shipment size
is wrong.

### Tuning `iou_thresh` — ours is `0.65` in `poc/models/base.py`

| Value | Behaviour | The failure it causes |
|-------|-----------|-----------------------|
| **0.3** (aggressive) | deletes anything moderately overlapping | **UNDER-count.** Two dining chairs side by side, one partly behind the other, get merged into one chair |
| **0.65** (ours) | middle ground | — |
| **0.9** (permissive) | only deletes near-identical boxes | **OVER-count.** Duplicates survive; one sofa reported as three |

**How to diagnose.** Look at the Stage 2 visualisation:

- *Six dining chairs in the photo, three reported* → NMS is too aggressive. **Raise** it.
- *One sofa, three boxes stacked on it* → NMS is too permissive. **Lower** it.

Chairs around a dining table are the hard case — from a camera angle they overlap heavily
while being genuinely separate objects. If our footage is chair-heavy, expect to raise this
toward 0.75.

## Threshold vs. NMS — they are not the same thing

Easy to confuse, and they fail in opposite directions:

```
  confidence threshold  ->  "is there anything here at all?"    filters by SCORE
  NMS IoU threshold     ->  "have I already reported this?"     filters by OVERLAP
```

A low confidence threshold with high NMS gives you **lots of duplicated junk**. A high
confidence threshold with low NMS gives you **too few, over-merged** detections.

## Open vs closed vocabulary

**Closed** — the model has a fixed list of output slots (YOLO: 80 COCO classes). Cannot be
asked for anything else.
**Open** — the model takes *your words* as input and matches them against image regions. Add
a class by editing a text file.

---

# Lesson 1 · Grounding DINO

**File:** `poc/models/detect_grounding_dino.py` · Apache 2.0 · boxes, no masks

## The intuition

It reads your shopping list and scans the room **at the same time, repeatedly** — glance at
the list, glance at the room, glance back — until each word has been matched to something
visible.

```
   "sofa. wardrobe. door."        the photo
            |                         |
       [text encoder]           [image encoder]
            |                         |
            +----> FUSE <-------------+     text reshapes how it
            |        |                |     looks at the image, and
            +------- + ---------------+     vice versa. Repeatedly.
                     |
                     v
            pick promising regions
                     |
                     v
         900 "clipboards", each filled in with
         one object or nothing
                     |
                     v
         [ sofa 0.71  box(412,288,901,604) ]
```

Two things follow from that picture, and they explain most of its behaviour:

1. **Knowing what you're looking for changes how it looks.** That's why it's accurate.
2. **All of that runs on every frame.** That's why it's ~20× slower than YOLO-World.

## How the label is produced — this explains most surprises

It does **not** output a class ID. Our 26 phrases go in as **one string**:

```
"sofa. armchair. coffee table. ... door."
```

For each box it scores similarity against **every word in that string**, and the label is
whichever span scored highest.

Consequences you will actually hit:

- **Phrasing matters.** "sofa" and "settee" are different tokens and score differently.
  This is why our vocabulary is a fixed curated list, not free text.
- **Multi-word phrases split.** "coffee table" is two tokens; the model may match only
  "table". That's why `detect_vocab.json` maps a phrase to *candidate size classes* rather
  than assuming one-to-one.
- **Vocabulary length costs accuracy.** More phrases = more chances to confuse. Adding 40
  more words will make the existing ones slightly worse.

## The knobs

All in `CFG` in the notebook, or as constructor args to `GroundingDinoDetector`.

### `det_box_th` — default `0.30`

*"How sure must it be that an object exists here?"*

| Value | Effect | When to use it |
|-------|--------|----------------|
| **0.15** | finds much more, invents a lot | you're missing furniture and a human reviews everything |
| **0.30** (ours) | balanced | starting point |
| **0.50** | only confident detections | you're drowning in false positives |

**Which way should we err?** **Lower.** For a survey, a *missed* wardrobe is a 100% error
on that item, while a false positive gets deleted by the reviewer in two seconds. Recall
beats precision here. If in doubt, drop this to 0.20 before touching anything else.

### `det_txt_th` — default `0.25`

*"How well must a word match before it becomes the label?"*

This one controls **naming**, not existence. Lower it and boxes get labelled with words
that only loosely match — a chest of drawers becomes a "sideboard".

| Value | Effect | Symptom that says "change me" |
|-------|--------|-------------------------------|
| **0.15** | loose naming | objects labelled as the wrong-but-similar thing |
| **0.25** (ours) | balanced | — |
| **0.40** | strict naming | boxes found but labelled as nothing, so dropped entirely |

**Diagnostic:** if Stage 2 shows lots of boxes but the histogram of labels is thin, this is
too high. If the counts look right but the *classes* are wrong, it's too low.

### `iou_thresh` in `nms()` — default `0.65`

See Lesson 0. Grounding DINO is DETR-based so it technically doesn't need NMS — we apply it
anyway so every detector in the comparison is post-processed the same way. Otherwise you'd
be comparing default post-processing settings, not models.

### `resize_long_edge` — default `1024`

Not a model parameter, but it changes detection more than either threshold.

| Value | Effect |
|-------|--------|
| 640 | fast, misses small items (lamps, boxes, photo frames) |
| **1024** (ours) | balanced |
| 1536 | finds small items, ~2× slower, more false positives |

**Diagnostic:** if you're consistently missing *small* objects, raise this before lowering
`det_box_th`. Resolution is usually the real cause.

## Tuning order

When output looks wrong, work through it in this order — earlier items have bigger effects:

```
  1. resize_long_edge   are small objects even visible?
  2. det_box_th         are objects being found at all?
  3. iou_thresh         are counts right? (duplicates or merges)
  4. det_txt_th         are the labels right?
  5. detect_vocab.json  is the word even in the list?
```

Step 5 catches more problems than people expect. If a rolled rug is never detected, check
it's in the vocabulary before assuming the model is bad.

## Symptom → cause table

| What you see | Likely cause | Fix |
|--------------|--------------|-----|
| Small items (lamps, boxes) missed | resolution too low | `resize_long_edge` → 1536 |
| One sofa, three boxes | NMS too permissive | `iou_thresh` → 0.5 |
| Six chairs, two reported | NMS too aggressive | `iou_thresh` → 0.8 |
| Furniture not found at all | confidence too high | `det_box_th` → 0.20 |
| Right boxes, wrong names | text threshold too low | `det_txt_th` → 0.35 |
| Boxes found, then vanish | text threshold too high | `det_txt_th` → 0.20 |
| One class never appears | not in the vocabulary | edit `detect_vocab.json` |
| Everything slow | it's the model, not a setting | try `rtdetr`, or use a GPU |

---

*Lessons 2–8 to follow. Say "next" for OWLv2, which makes the opposite design choice and
is a useful contrast.*
