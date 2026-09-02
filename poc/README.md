# POC — can a model identify room contents and how much space they take?

> **FILE PURPOSE** — Quick start. How to get the pipeline running.
> For *how it works* read `../ARCHITECTURE.md`. For *what's still undecided* read
> `../GAPS.md`. For *what's actually built* read `../STATUS.md`.

One notebook, two competing methods, scored against hand measurements.

## The two methods

| | Method | The AI's job | Mental model |
|---|---|---|---|
| **A** | **Recognise & Look Up** | name things correctly | surveyor with a clipboard |
| **B** | **Measure & Compute** | measure things correctly | surveyor with a tape measure |

**Method A** looks at the sofa, decides *"3-seater"*, and reads `1.42 m³` off
`cube_table.json`. It never measures anything.

**Method B** works out the sofa is 2.08 x 0.91 x 0.86 m and multiplies. It never looks
anything up.

Every product in this market uses Method A and none of them measure objects. This POC
checks that on our own footage rather than taking it on trust. `ARCHITECTURE.md` §7 walks
one sofa through both routes if that helps.

## Setup

```bash
./setup.sh                               # ~20 min, pulls ~2.5GB of torch
export ANTHROPIC_API_KEY=sk-ant-...      # Method A only; Method B runs without it
```

Drop input into `data/input/` — images (`.jpg .png`) or video (`.mp4 .mov`).

```bash
source .venv/bin/activate && jupyter lab pipeline.ipynb
```

Pick the **NX Survey POC** kernel.

## Before the numbers mean anything

Copy `ground_truth_template.csv` to `ground_truth.csv` and fill it in for the rooms you
shoot — laser measure, one row per item. Without it the notebook still prints volumes but
nothing tells you whether they are right. That is gap **A1**, and it is roughly half a day
for 3–5 rooms.

`ground_truth.csv` is gitignored because it can carry property addresses.

## Stages

| Stage | Does | Model |
|-------|------|-------|
| 1 Ingest | video/image → sharp keyframes | opencv |
| 2 Detect | find and **count** objects | Grounding DINO |
| 3 Depth | 3D position per pixel, in metres | MoGe-2 |
| 4 **Scale anchor** | correct scale from a 1981 mm door | — |
| 5 **Method B** | measure each object | — |
| 6 **Method A** | name each object's size class | Claude Sonnet 5 |
| 7 Dedup | one sofa, not twenty | — |
| 8 Aggregate | three volume figures | — |
| 9 Evaluate | bias and spread vs ground truth | — |
| A Appendix | how much precision is needed — **runs with no data** | — |

## What to look at first

Stage 9 prints **bias** separately from **absolute error**. Bias is the one that matters —
consistently 15% low is fixable with one multiplier; randomly ±15% is not.

Then run stage 4 twice, `use_scale_anchor` True then False, and compare. **That delta is
the most valuable number the POC produces.**

Try the appendix before you have any footage — it predicts what precision Method B needs.
