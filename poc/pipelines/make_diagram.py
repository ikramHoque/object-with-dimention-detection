"""
FILE PURPOSE
    Generates DIAGRAM.md for a pipeline: a system-design view of what happens to a
    photograph, stage by stage, naming the actual models that pipeline uses.

WHO IT IS FOR
    Someone who has to understand or approve the design without reading the code — a
    team lead, or you in three months. It explains the INTUITION at each step: not
    only what the stage does but why it exists and how it fails.

HOW TO RUN
    python -m poc.pipelines.make_diagram grounding_dino__moge2__sonnet5
    python -m poc.pipelines.make_diagram --all

WHY GENERATED
    Fourteen pipelines share one design and differ only in which models fill the
    slots. A hand-written diagram per folder would drift the first time a stage
    changed, and a wrong diagram is worse than none — it gets believed. One source,
    regenerated.

    The diagrams use Mermaid, which GitHub renders inline, so the file is readable in
    a browser without any tooling.

USED BY  you, when the design needs explaining
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def build(cfg) -> str:
    name = cfg.NAME
    det, dep = cfg.DETECTOR, cfg.DEPTH
    seg = getattr(cfg, "SEGMENTER", None)
    cls = getattr(cfg, "CLASSIFIER", None)
    noanchor = getattr(cfg, "NO_ANCHOR", False)
    ship = getattr(cfg, "SHIPPABLE", True)

    from poc.models import registry
    def lic(k):
        return registry.info(k).licence if k else "-"
    def disp(k):
        return registry.info(k).display if k else "-"

    # ---------- the data-flow diagram ----------
    seg_node = (f'    S2 -->|"boxes"| SEG["<b>Stage 2b · Segment</b><br/>{seg}<br/>'
                f'<i>box to pixel mask</i>"]\n'
                f'    SEG -->|"masks"| S4\n') if seg else ""
    seg_from = "SEG" if seg else "S2"

    anchor_body = (
        '    S4["<b>Stage 4 · Scale anchor</b><br/>door_scale()<br/>'
        '<i>a UK door is 1.981 m</i>"]\n'
        '    S4 -->|"one multiplier<br/>e.g. x0.8589"| S5\n') if not noanchor else (
        '    S4["<b>Stage 4 · Anchor DISABLED</b><br/>scale = 1.0<br/>'
        '<i>this run is the ablation</i>"]\n'
        '    S4 -->|"raw depth, uncorrected"| S5\n')

    cls_block = (
        f'    S1 --> S6["<b>Stage 6 · Recognise</b><br/>{cls}<br/>'
        f'<i>picks a SIZE CLASS, never measures</i>"]\n'
        f'    S6 -->|"size classes + counts<br/><b>Method A</b>"| S7\n') if cls else (
        '    S6["<b>Stage 6 · skipped</b><br/>no classifier<br/>'
        '<i>Method B alone, no API key</i>"]\n'
        '    S6 -.-> S7\n')

    flow = f"""```mermaid
flowchart TD
    IN[/"photo · video · folder of stills<br/><code>data/input/</code>"/]
    IN --> S1["<b>Stage 1 · Ingest</b><br/>load_keyframes()<br/><i>resize, drop blurry frames</i>"]

    S1 -->|"frames, long edge 1024px"| S2["<b>Stage 2 · Detect</b><br/>{det}<br/><i>finds and names objects</i>"]
    S1 -->|"the WHOLE frame, never crops"| S3["<b>Stage 3 · Depth</b><br/>{dep}<br/><i>metres per pixel</i>"]

{seg_node}    {seg_from} -->|"boxes + labels"| S4
    S3 -->|"(X,Y,Z) in metres"| S4
{anchor_body}
    S5["<b>Stage 5 · Measure</b><br/>extent() · resolve_dims()<br/><i>PCA footprint to W x D x H</i>"]
    S5 -->|"dimensions + m3<br/><b>Method B</b>"| S7

{cls_block}
    S7["<b>Stages 7-8 · Dedup + total</b><br/>dedup_measured()<br/>vol_from_inventory()<br/><i>the same sofa counted once</i>"]
    S7 --> CUBE[("cube_table.json<br/>42 size classes<br/>-> PACKED volume")]
    CUBE --> S9["<b>Stage 9 · Score</b><br/>score_inventory()<br/><i>silent without ground truth</i>"]

    S9 --> OUT1[/"results/&lt;run&gt;.json"/]
    S9 --> OUT2[/"frame_000.jpg<br/>boxes + dimensions"/]
    S9 --> OUT3[/"items.csv<br/>for the surveyor"/]

    style S4 fill:#0A6560,color:#fff
    style CUBE fill:#4FB2AA,color:#000
    style S5 fill:#1E8D85,color:#fff
    style S9 fill:#eeeeee,color:#000
```"""

    # ---------- call order ----------
    seq_seg = f"    R->>SEG: refine(frame, boxes)\n    SEG-->>R: masks\n" if seg else ""
    seq_cls = (f"    R->>C: classify(frame, class_names)\n"
               f"    C-->>R: size classes + counts (Method A)\n") if cls else \
              "    Note over R: no classifier — Method B only\n"
    seq = f"""```mermaid
sequenceDiagram
    autonumber
    actor U as You
    participant R as run.py / notebook
    participant D as {det}
    participant SEG as {seg or 'segmenter'}
    participant M as {dep}
    participant P as pipeline.py
    participant C as {cls or 'classifier'}
    participant O as results/

    U->>R: --input my_room.jpg
    R->>P: load_keyframes(src)
    P-->>R: frames (blurry ones dropped)
    R->>D: detect(frame, 26 prompts)
    D-->>R: boxes + matched text
{seq_seg}    R->>M: infer(frame)
    M-->>R: (X,Y,Z) metres, mask, intrinsics
    R->>P: door_scale(dets, depth)
    P-->>R: scale multiplier
    loop each detected object
        R->>P: extent(depth, det, scale)
        R->>P: resolve_dims(extent, classes, allowed)
        P-->>R: W x D x H + size class
    end
{seq_cls}    R->>P: dedup + vol_from_inventory()
    R->>O: JSON + annotated frame + items.csv
    O-->>U: a volume, and the evidence for it
```"""

    ship_note = "" if ship else f"""
> ## ⚠ THIS PIPELINE CANNOT SHIP
>
> It depends on a model whose licence forbids it. Read
> `ALERT_DO_NOT_SHIP.md` in this folder before running anything. It exists to measure
> **what staying licence-clean costs us** — a better score here is an argument for
> buying a licence, not permission to ship.
"""

    return f"""# {name} — system design

> **FILE PURPOSE** — How a photograph becomes a volume in this pipeline, stage by
> stage, with the intuition behind each step. Written to be read by someone who will
> not read the code. Generated by `poc/pipelines/make_diagram.py` — do not hand-edit.
{ship_note}
**The question this pipeline answers:** {getattr(cfg, 'QUESTION', '—')}

| slot | model | licence |
|---|---|---|
| detector | `{det}` — {disp(det)} | {lic(det)} |
| depth | `{dep}` — {disp(dep)} | {lic(dep)} |
| segmenter | {f'`{seg}` — {disp(seg)}' if seg else '_none — measures the whole box, background included_'} | {lic(seg)} |
| classifier | {f'`{cls}` — {disp(cls)}' if cls else '_none — Method B only, no API key needed_'} | {lic(cls)} |
| scale anchor | {'**DISABLED** — this run is the ablation' if noanchor else 'door, 1.981 m'} | — |

---

## 1 · The idea in one paragraph

A photograph gives you pixels. A quote needs cubic metres. Getting from one to the
other takes two independent routes, because each fails where the other works.
**Method B measures**: a depth model turns pixels into metres, and geometry turns
metres into width, depth and height. **Method A recognises**: a vision model names
each object's *size class*, and a lookup table turns that class into the volume the
object occupies **once packed** — which is not the same as its measured volume, since
a rug ships rolled. When the two agree you can quote; when they disagree you have
found a room that needs a human. That, not a single accuracy number, is the product.

---

## 2 · The data flow

{flow}

---

## 3 · Step by step, and why each step exists

| # | Stage | In | Out | Why it exists |
|---|---|---|---|---|
| 1 | **Ingest** `load_keyframes()` | your file | frames at 1024 px, blurry ones dropped | Everything downstream is in **pixels** until stage 4, so frames must be a consistent size. Blur wrecks both detection and depth, so it is filtered before a model ever sees it. |
| 2 | **Detect** `{det}` | frame + 26 prompts | boxes + matched text | An open-vocabulary detector returns *matched text*, not a class id, so the label is normalised to one prompt. Where it cannot pick one it returns **no** label rather than guessing — a wrong label forces the wrong size class whatever the geometry says. |
{f"| 2b | **Segment** `{seg}` | boxes | one mask per object | A box around a sofa also contains floor and wall, and those pixels get measured as part of the sofa. Masks measure the object only. Measured effect: door error 16.5% -> 6.8%. |{chr(10)}" if seg else ""}| 3 | **Depth** `{dep}` | the **whole** frame | (X,Y,Z) in metres per pixel | Run on the full frame, never the crops — these models reason from scene context, and cropping destroys the geometry they use to infer scale. The metres are *claimed*: about 8% error. |
| 4 | **Scale anchor** `door_scale()` | boxes + depth | {'**nothing — disabled in this run**' if noanchor else 'one multiplier'} | {'This run deliberately turns the anchor OFF, so its result minus the baseline is exactly what the anchor is worth.' if noanchor else 'The highest-leverage stage. Depth error is **one multiplier on the whole room**, not noise that averages out — and volume goes as length cubed, so 16% on length is 58% on volume. A door is a known 1.981 m, so it calibrates everything at once.'} |
| 5 | **Measure** `extent()` `resolve_dims()` | box + depth + scale | W x D x H + m3 | PCA on the horizontal footprint finds the object's *own* axes — a sofa at 30° is 2.2 m long, not 2.5 m of diagonal. A camera never sees the back of a wardrobe, so when depth is unobservable a class prior is substituted and `depth_source` says so. |
| 6 | **Recognise** {f'`{cls}`' if cls else '_skipped_'} | the frame | size classes + counts | {'Names the SIZE CLASS, never measures. "sofa" does not set a volume — sofa_2_seat 0.99 m3 vs sofa_sectional 2.27 m3 is a 2.3x spread, and choosing between them *is* the quote. Not used for counting: VLM counting accuracy is ~0.53 and under-counts.' if cls else 'No classifier in this pipeline, so there is no Method A and no cross-check — and no API key needed. This is the run that answers "how good is pure geometry alone?"'} |
| 7-8 | **Dedup + total** | all rows | one inventory + volumes | The same sofa appears in six frames and must count once. Then each object is replaced by its cube-table class, because **packed volume is a property of the class**, not of the measurement. |
| 9 | **Score** `score_inventory()` | inventory + ground truth | bias, error, precision, recall | **Silent unless `poc/ground_truth.csv` exists.** Without hand-measured rooms the pipeline can report what it believes, never whether it is right. Gap **A1**. |

---

## 4 · Call order

{seq}

---

## 5 · Where this pipeline goes wrong — what to watch

| Symptom in the output | What it means | What to do |
|---|---|---|
| `labels: N ambiguous` | `text_threshold` too LOW — prompts fused into one span | raise `TEXT_TH` in `config.py` |
| `labels: N empty` | `text_threshold` too HIGH — no token cleared it | lower `TEXT_TH` |
| `SCALE=1.0  [no door found]` | **no calibration happened** — raw ~8% depth error, ~26% on volume, flows into the quote | get a door in frame, or accept the error explicitly |
| `class_prior_pct` above 60 | Method B is mostly reading the cube table, so it is no longer independent of Method A | their agreement stops being evidence |
| raw m3 far above class-snapped | boxes contain wall and floor | {'masks are on; check they are not too tight' if seg else 'add a segmenter — `grounding_dino__moge2__sam2__sonnet5`'} |
| a 1.7 m deep sofa in `items.csv` | over-measurement: background inside the box | look at `frame_000.jpg` — this is exactly what it is for |

**Read the annotated frame, not just the JSON.** Every real bug found so far — a 19%
over-estimate from a mislabelled coffee table, a door-anchor percentile error, an
unnamed-duplicate double count — looked perfectly plausible in the numbers and was
obvious the moment the boxes were drawn on the photograph.

---

## 6 · The honest limitation

This pipeline reports what it **believes**. Nothing in it establishes whether it is
**right**. `HomeObjects-3K` scores detection only; `NYU Depth V2` can score depth; but
neither carries the *packed volume* of a real room, and that is the number NX quotes.

Until 3-5 rooms are measured by hand into `poc/ground_truth.csv`, every figure here is
a demonstration, not a measurement. That is gap **A1** in `GAPS.md`, and no amount of
further modelling substitutes for the tape measure.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a pipeline's DIAGRAM.md")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    names = ([q.name for q in sorted(HERE.iterdir()) if (q / "config.py").is_file()]
             if a.all else [a.name])
    if not names or names == [None]:
        print("name a pipeline, or --all")
        return 1
    for n in names:
        cfg = importlib.import_module(f"poc.pipelines.{n}.config")
        out = HERE / n / "DIAGRAM.md"
        out.write_text(build(cfg))
        print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(out.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
