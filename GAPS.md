# NX Pre-Move Survey AI — Open Gaps

> **FILE PURPOSE** — Everything still undecided or unbuilt, with a recommendation for each
> so it closes with a decision rather than a discussion. Tick items off as they resolve.
>
> Map: `ARCHITECTURE.md` · Build state: `STATUS.md`

A working checklist. Each gap has a recommendation so it can be closed with a decision
rather than a discussion. Tick them off as they are resolved.

**Grouping is by *when* it must be answered, not by how hard it is.**

- **Group A** — decide before writing pipeline code. These change what we build.
- **Group B** — solve inside the POC. These are engineering problems.
- **Group C** — deliberately parked until the POC reports. Listed so nothing is forgotten.

---

## Summary

| # | Gap | Who decides | Status |
|---|-----|-------------|--------|
| **A1** | Nothing to measure accuracy against | Us + NX | ☐ open |
| **A2** | "Accuracy" is still undefined | NX | ☐ open |
| **A3** | No cube table or size-class list | NX | ☐ open |
| **A4** | Two possible methods, no choice made | Us | ☐ open |
| **A5** | How the system learns what a metre is | Us | ☐ open |
| **B1** | Same object counted in many frames | Us | ☐ open |
| **B2** | Some best models are licence-blocked | Us | ☐ open |
| **B3** | Objects hidden behind other objects | Us | ☐ open |
| **B4** | Packed size ≠ real size | Us + NX | ☐ open |
| **B5** | Which video frames to actually use | Us | ☐ open |
| **B6** | No GPU on the dev machine | Us | ☐ open |
| **B7** | Every room is different | Us | ☐ open |
| **B8** | Cameras only see the front of things | Us | ☐ open |
| **B9** | Ultralytics YOLO is AGPL-licensed | Us + NX | ☐ open |
| **C1** | Contents of closed cupboards | — | ☐ parked |
| **C2** | Rooms the customer never photographs | — | ☐ parked |
| **C3** | Carton / box counts | — | ☐ parked |
| **C4** | Dismantling and fragile packing | — | ☐ parked |
| **C5** | GDPR, DPIA, data residency | NX Legal | ☐ parked |
| **C6** | Security of the photo archive | NX Security | ☐ parked |
| **C7** | Build vs buy never answered | NX | ☐ parked |
| **C8** | Which back-office system to feed | NX | ☐ parked |
| **C9** | Self-pack vs full-pack | NX | ☐ parked |
| **C10** | No baseline for NX's current process | NX | ☐ parked |

---

# Group A — Decide before writing pipeline code

### A1 · Nothing to measure accuracy against
**The gap.** We plan to test whether the model gets dimensions right, but we have no
correct answers to compare against. Right now there is no way to tell a good result
from a bad one.

**Why it matters.** Without this the POC produces demos, not findings. The whole point
of the exercise — "is this feasible and how accurate is it" — cannot be answered.

**Options.**
- (a) Hand-measure a few real rooms ourselves with a laser measure.
- (b) Ask NX for completed jobs with photos plus the actual loaded volume.
- (c) Skip it and eyeball the output.

**Recommendation.** Do (a) now and request (b) in parallel. Option (a) is half a day:
pick 3–5 rooms, list every item, record its size class and its width/depth/height.
That is enough to score the POC. Option (b) is what scores the real product later.
Option (c) is how projects like this quietly fail.

**Blocks.** Every accuracy number in the POC.

---

### A2 · "Accuracy" is still undefined
**The gap.** NX asked for 5%. Five percent of what — total volume, one room, one item?
Averaged over jobs, or true for 95 of 100 jobs? Measured against a surveyor's estimate
or against what actually went on the truck?

**Why it matters.** These read the same in a meeting and differ by a factor of three in
a contract.

**Recommendation.** For the POC, define our own metric set and report all of it:
per-item class accuracy, count error, dimension error, and total volume error split
into **bias** (are we consistently high or low) and **spread** (how much we vary).
Take that to NX as the basis for a definition rather than asking them to define it cold.

---

### A3 · No cube table or size-class list
**The gap.** Both working products in this market assign volume by looking a classified
item up in a reference table. We do not have NX's table, and published industry tables
disagree with each other — the same item carries different values, and "wardrobe" means
a piece of furniture on one sheet and a hanging carton on another.

**Why it matters.** This is the backbone of the whole system. It also means 10–20%
disagreement can enter before any AI is involved.

**Recommendation.** Ask NX for their cube sheet and carton specifications as the first
formal request of the project. Meanwhile bootstrap a placeholder table so the POC can
run — one is included in `poc/cube_table.json`, clearly marked as provisional.

**Blocks.** Method A of the POC produces numbers that cannot be trusted until this is real.

---

### A4 · Two possible methods, no choice made
**The gap.** There are two ways to get a volume, and we have not chosen:
- **Classify and look up** — decide "this is a 3-seater sofa", read the volume off a table.
- **Measure geometrically** — reconstruct the object in 3D and compute its volume.

**Why it matters.** They need different engineering. Both incumbents use the first and
neither measures objects. But we have not verified that on our own data.

**Recommendation.** Build both in the POC and score them against A1's measurements.
Costs roughly 20% more than building one and settles the architecture with evidence.
Expect classify-and-look-up to win; if measurement wins we have found something the
incumbents have not.

**Already measured.** The notebook appendix simulates this without any input data. At the
depth model's published 8% scale error, mapping a measurement onto the nearest size class
is right about **90%** of the time *if* the detector first narrows the candidates, and only
**72%** if it searches the whole cube table. So Method B is more viable than expected — but
only as *detector plus geometry*, never geometry alone.

---

### A5 · How the system learns what a metre is
**The gap.** A photograph has no scale. A sofa and a doll's sofa are identical images.
Something must supply absolute size.

**Why it matters.** This is the error that hurts most, because it is a single multiplier
on everything. If a room is read 8% too large, every object in it is 8% too large — and
volume errors are roughly three times linear errors, so 8% becomes about 26%. Averaging
over many objects does not help, because there is only one mistake, not many.

**Options.**
- (a) Trust the depth model alone.
- (b) Anchor on something of known size in the frame — UK internal doors are 1981 mm.
- (c) Ask the person to confirm the room size once.
- (d) Use hardware depth (iPhone LiDAR) — but that needs a native app.

**Recommendation.** Build (b) and (c), and make them switchable so we can measure what
they are worth. Do not build only (a) — it will produce a bad number and make the POC
look like a failure when it is really just the worst configuration.

**Already measured.** Class-assignment accuracy against scale error, from the notebook
appendix: **0% → 99%, 5% → 95%, 8% → 90%, 15% → 78%**. Roughly one point of class accuracy
per point of scale error in the range that matters. That is the return on the anchor work,
and it is why A5 outranks model selection.

The pairs that break first are the ones separated by the smallest margin — double vs king
bed, double vs king mattress, armchair vs recliner. For those, **asking the customer one
size question beats any amount of better geometry**, which is also how the incumbents
appear to handle "queen vs king mattress".

---

# Group B — Solve inside the POC

### B1 · Same object counted in many frames
**The gap.** One sofa appears in twenty video frames. Naive pipelines report twenty sofas.

**Why it matters.** Counting errors go straight into the volume. This is the most common
way a pipeline like this produces a wildly wrong number.

**Recommendation.** Start with the simple rule — for each item type, take the highest
count seen in any single frame — then improve to tracking objects between frames.
The simple rule is in the notebook and is good enough to get a first reading.

---

### B2 · Some best models are licence-blocked
**The gap.** UniDepthV2 posts the best indoor depth numbers in the field and is
CC BY-NC — non-commercial. The headline Depth Anything 3 variants are the same.

**Why it matters.** Building a prototype on a model we cannot ship means rewriting it.

**Recommendation.** Use only the commercially clean stack from the start:
**MoGe-2** (MIT) for depth, **DA3Metric-Large** (Apache 2.0) if multi-view is needed,
**SAM 3** (Meta licence, commercial use permitted) for segmentation. Have counsel confirm
the specific weight files before launch, not before the POC.

---

### B3 · Objects hidden behind other objects
**The gap.** A dining chair half behind a table. A box under a bed. Only part is visible,
or none of it is.

**Recommendation.** Accept it in the POC and record it as a known error source. This is
what the guided capture and the human review exist to catch. Do not try to solve it now.

---

### B4 · Packed size ≠ real size
**The gap.** We are planning to output real-world dimensions. But a removal charges for
*packed* volume. A vacuum cleaner ships at roughly its original box size. A desk may be
dismantled and take much less room.

**Why it matters.** Even a perfectly measured object can give the wrong answer to the
business question. This is a second argument for the lookup table, which can hold packed
volume directly.

**Recommendation.** Have the cube table store packed volume, and treat measured
dimensions as an input to classification rather than as the final answer.

---

### B5 · Which video frames to actually use
**The gap.** A one-minute room video is ~1,800 frames. Most are blurred, redundant, or
pointed at the floor.

**Recommendation.** Sample at a fixed interval, then drop frames that fail a sharpness
and brightness check. Implemented in the notebook so we can tune it against real footage.

---

### B6 · No GPU on the dev machine
**The gap.** The dev machine is an M1 with 16 GB and no CUDA. MoGe-2 is a 326M-parameter
model.

**Why it matters.** It will run on Apple's MPS backend but slowly. Fine for tens of
frames, not for a hundred homes.

**Recommendation.** Run the POC locally on the small model variants; move to a rented
GPU when we start batch-scoring real jobs. Compute cost is not a constraint — roughly
$0.60 per survey all-in — so this is a convenience question, not a budget one.

---

### B7 · Every room is different
**The gap.** Lighting, clutter, room size, property age and camera all vary enormously.
A result from one room says very little.

**Recommendation.** Restrict the POC to two room types — bedroom and living room — across
3–5 different properties. Fewer variables, and the numbers start to mean something.

---

### B8 · Cameras only see the front of things
**The gap.** Found while testing the measurement code, not predicted. A camera never
sees the back of a wardrobe. The depth model returns only the *visible surface*, so a
flat-fronted object's point cloud is a thin sheet with no measurable depth — and the
computed volume collapses toward zero.

**Why it matters.** This is a hard limit of single-view geometry, not noise, and it
affects most furniture: wardrobes, chests, bookcases, fridges, anything against a wall.
Method B therefore **systematically under-measures depth** on exactly the items that
carry the most volume.

**What we did.** `resolve_dims()` detects the degenerate case (second principal axis
carries under 12% of the variance), matches the object on the two dimensions that *were*
observed — width and height — and adopts that class's typical depth from the cube table.
Verified: a wardrobe-shaped flat surface resolves to `wardrobe_double` with a 0.60 m
depth instead of a 0.00 m one.

**The catch worth watching.** Every run records `class_prior_pct`. If that is high,
**Method B is getting its depth from the cube table, which is Method A's data** — so the
two methods are no longer independent and the comparison means less than it appears.
The runner prints a warning above 60%.

**Open question for the POC.** Measure that percentage on real footage. If it is high,
the honest conclusion is that pure geometric measurement is not viable from single-view
capture, which strengthens the case for Method A and for multi-view depth.

---

### B9 · Ultralytics YOLO is AGPL-licensed
**The gap.** YOLO-World and YOLOE are the fastest open-vocabulary detectors available —
roughly 20x faster than Grounding DINO at a fifth of the size. Both are normally run
through the Ultralytics package, which is **AGPL-3.0**.

**Why it matters.** Per Ultralytics' own published position, *any* use of their models —
explicitly including R&D inside a company, commercial or not — requires a paid Enterprise
Licence unless the entire project is open-sourced under AGPL-3.0. AGPL is stricter than
GPL and reaches SaaS deployment, so exposing it only as an API is not an escape.

**Options.**
- (a) Don't use it. Grounding DINO and OWLv2 are Apache 2.0 and open-vocabulary.
- (b) Use RT-DETR (Apache 2.0) for speed, accepting a fixed vocabulary.
- (c) Buy the Enterprise Licence, if the speed proves worth it.
- (d) Run it for comparison only, and never ship it.

**Recommendation.** Default to (a). Registered `commercial_ok=False`, so the runner
refuses these models unless `--allow-noncommercial` is passed, and tags such results
`shippable=false`. Decide (c) only if the sweep shows the speed advantage is large AND
NX's volume makes it matter. Get NX's legal view before even the comparison run, since
Ultralytics' position covers internal R&D.

---

# Group C — Parked until the POC reports

These are real and already understood. They are listed so they are not rediscovered late.

| Gap | Note |
|-----|------|
| **C1 · Contents of closed cupboards** | The largest error source in the finished product, and not a vision problem at all — the information is not in the photograph. Plan is to measure storage frontage, apply NX's own density factors, and ask the customer one question per unit. Out of POC scope. |
| **C2 · Rooms never photographed** | A missed garage is a 100% error on that room. Solved with a coverage checklist and a completion gate, not with a model. |
| **C3 · Carton / box counts** | Follows directly once C1 and the cube table exist. |
| **C4 · Dismantling and fragile packing** | Rules and policy, not perception. Belongs in an editable table. |
| **C5 · GDPR, DPIA, data residency** | Photos of homes are personal data. Needed before any customer touches this. One question is urgent even now: may customer images be processed by models hosted outside the UK/EU? It constrains the architecture. |
| **C6 · Security of the photo archive** | A house inventory plus a move date is a target list. Treat as high-sensitivity from day one. |
| **C7 · Build vs buy never answered** | Incumbents sell this from about $20 a survey. Worth a direct comparison before committing to a build. |
| **C8 · Which back-office system to feed** | Unknown to us. Commonly a large source of unplanned effort. |
| **C9 · Self-pack vs full-pack** | Changes carton counts materially. Competitors model it explicitly as CP and PBO. Cheap to handle early, expensive to retrofit. |
| **C10 · No baseline for NX's current process** | We do not know their current error, survey volume, or cost per survey. Every target needs that denominator. |

---

## What closing Group A unlocks

Once A1–A5 are decided, the POC can give a straight answer to the original question:
**can a model identify what is in a room, and how much space it will take?**

Everything in Group B is then ordinary engineering, and Group C becomes a product
roadmap rather than a set of unknowns.
