# `poc/` — the working code

```
pipelines/          one folder per pipeline: config, CLI, notebook, input, results
  grounding_dino__moge2/
  grounding_dino__moge2__sam2/
  scaffold.py       create a new pipeline folder
  make_notebook.py  generate a pipeline's step-by-step notebook

models/             13 model adapters behind one interface, plus the licence gate
runner/             the shared machinery every pipeline calls
  pipeline.py         the nine stages — all the measurement maths
  run_combination.py  the orchestrator
  report.py           annotated frames, item tables, CSV
  run_dataset_eval.py detection scoring against HomeObjects-3K
  run_sweep.py        the OFAT experiment matrix
  compare.py          rank finished runs
datasets/           HomeObjects-3K loader (reads the zip with OpenCV, so no AGPL
                    code enters the project)
rnd/                quarantine for models that cannot ship — measurement only
colab/              GPU notebooks

cube_table.json     42 size classes with volumes. PLACEHOLDER — not NX's own sheet
detect_vocab.json   26 prompts, and which size classes each one permits
combinations.json   the 14-run experiment plan for run_sweep
ground_truth_template.csv  copy to ground_truth.csv once rooms are measured
```

## Run something

```bash
./setup.sh                                        # once, ~20 min
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

cp my_room.jpg pipelines/grounding_dino__moge2/data/input/
python -m poc.pipelines.grounding_dino__moge2.run                 # one shot
jupyter lab pipelines/grounding_dino__moge2/notebook.ipynb        # stage by stage
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
| `../STATUS.md` | what exists, and what has actually been executed |
| `pipelines/README.md` | what belongs in a pipeline folder and what does not |

## The one thing to keep in mind

The pipeline reports what it **believes**, not what is **true**. There is no measured
volume ground truth yet, so stage 9 stays silent and no accuracy claim exists. That is
gap **A1**, and it needs a tape measure, not more code.
