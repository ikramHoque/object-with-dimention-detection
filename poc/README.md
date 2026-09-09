# `poc/` — the working code

```
pipelines/            WHAT to run. 14 folders, one per combination.
  baseline/             each folder holds:
    config.py             the models and thresholds — THE definition
    run.py                CLI:  python -m poc.pipelines.baseline.run
    notebook.ipynb        the same pipeline, one stage per cell (generated)
    data/input/           put YOUR photos, videos or room-folders here
    results/              JSON + annotated frame + items.csv land here
    README.md             what it tests and how to run it
  anchor_off/  det_owlv2/  det_sam3/  det_rtdetr/  mask_sam2/
  depth_da3/  cls_opus5/  cls_haiku45/  method_b_only/  best_guess_stack/
  NOSHIP_AGPL__det_yolo_world/     <- cannot ship. ALERT_DO_NOT_SHIP.md inside.
  NOSHIP_AGPL__det_yoloe/             Evaluation only, to measure what
  NOSHIP_NC__depth_unidepth_ceiling/  staying licence-clean costs us.

  scaffold.py         create a new pipeline folder
  make_notebook.py    generate a pipeline's notebook (--all regenerates every one)
  _runner.py          config.py -> the orchestrator
  README.md           all 14 combinations, and what belongs in a folder

runner/               HOW it runs. Shared by every pipeline — never copied.
  pipeline.py           the nine stages: ALL the measurement maths
                        load_keyframes, door_scale, extent, resolve_dims,
                        allowed_classes, nearest_class, dedup, volume, score
  run_combination.py    the orchestrator: runs the stages, writes the outputs
  report.py             annotated frames, the item table, items.csv
  run_dataset_eval.py   detection scoring against HomeObjects-3K
  compare.py            rank finished runs by bias

models/               13 model adapters behind one interface
  base.py               Detection, DepthResult, nms, normalise_label, pick_device
  registry.py           model lookup, install probing, the LICENCE GATE
  detect_*.py           6 detectors    depth_*.py   3 depth models
  segment_sam2.py       1 segmenter    classify_claude.py  3 classifier sizes

datasets/homeobjects.py  loads HomeObjects-3K with OpenCV, deliberately not the
                         ultralytics package, so no AGPL code enters the project
rnd/                  quarantine for non-shippable research runs
colab/                GPU notebooks

cube_table.json       42 size classes -> packed volume. PLACEHOLDER, not the client's sheet
detect_vocab.json     26 prompts -> which size classes each prompt permits
ground_truth_template.csv  copy to ground_truth.csv once rooms are measured
data/homeobjects3k/   the downloaded dataset (411 MB, gitignored)
```

## The one structural rule

A **pipeline folder** says *what settings, what input, what output*.
**`runner/` and `models/`** say *how*.

The stage logic exists once, in `runner/pipeline.py`, and all 14 pipelines import it.
Fix a stage and every pipeline gets the fix. Copying the logic into each folder would
make the folders self-contained and wreck the codebase — 14 copies drift, and then
nobody can tell which is right. `poc/pipeline.ipynb` proved it: it carried its own
copy, was never executed, drifted into disagreeing with the CLI, and has been deleted.

The notebooks are generated for the same reason. Their narration is identical across
pipelines and differs only in which models it names, so it lives in one place —
`pipelines/make_notebook.py`. **Never edit a `notebook.ipynb` directly**; the next
`--all` overwrites it.

## Run something

```bash
./setup.sh                                        # once, ~20 min
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p pipelines/grounding_dino__moge2__sonnet5/data/input/lounge
cp ~/photos/lounge/*.jpg pipelines/grounding_dino__moge2__sonnet5/data/input/lounge/
python -m poc.pipelines.grounding_dino__moge2__sonnet5.run                 # one shot
jupyter lab pipelines/grounding_dino__moge2__sonnet5/notebook.ipynb        # stage by stage
```

Run from the repo root, not from here, so `poc.` imports resolve.

## Where things are written

Inside the pipeline's own `results/`: the JSON, an annotated `frame_000.jpg` with each
object labelled by class, W×D×H and m³, and `items.csv`.

## Reading order

| | |
|---|---|
| `../PIPELINE.md` | the nine stages, with the function and file behind each |
| `../RUNNING.md` | every way to run this, and troubleshooting |
| `../MODELS.md` | the catalogue, the licences, and how each model thinks |
| `../LEARN.md` | what NMS, IoU and the thresholds actually do |
| `../GAPS.md` | every open decision, **A1 first** |
| `../AI-CORE-DESIGN.md` | the AI core as a **shareable** design — abstracted, proposal-safe |
| `../SYSTEM-DESIGN.md` | the production build: modules, AWS, data model, CI/CD. **Internal** |
| `../STATUS.md` | what exists, and what has actually been executed |
| `pipelines/README.md` | what belongs in a pipeline folder and what does not |

## The one thing to keep in mind

The pipeline reports what it **believes**, not what is **true**. There is no measured
volume ground truth yet, so stage 9 stays silent and no accuracy claim exists. That is
gap **A1**, and it needs a tape measure, not more code.
