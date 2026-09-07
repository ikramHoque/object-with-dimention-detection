# Pipelines

One folder per pipeline. Each is self-contained for the things that belong to it —
**config, entry points, input, output** — and imports the stage logic from the shared
core.

```
poc/pipelines/
  grounding_dino__moge2/          <- a pipeline
    config.py                       which models, which thresholds
    run.py                          the CLI
    notebook.ipynb                  the same pipeline, one stage per cell
    data/input/                     YOUR photographs
    results/                        JSON + annotated frames + items.csv
    README.md
  scaffold.py                     <- create a new pipeline folder
  make_notebook.py                <- regenerate a pipeline's notebook
  _runner.py                      <- config -> run_combination plumbing
```

## Run one, two ways

```bash
cp my_room.jpg poc/pipelines/grounding_dino__moge2/data/input/

# one shot
python -m poc.pipelines.grounding_dino__moge2.run

# stage by stage
jupyter lab poc/pipelines/grounding_dino__moge2/notebook.ipynb
```

Both read `config.py` and both call the same shared functions, so they cannot give
different answers. Verified: the notebook and the CLI produce an identical inventory
total on the same photograph.

`--input` may be omitted when `data/input/` holds exactly one thing. With several it
lists them and stops rather than guessing.

## Add one

```bash
python -m poc.pipelines.scaffold --detector owlv2 --depth da3_metric
python -m poc.pipelines.make_notebook owlv2__da3_metric__nomask
```

The scaffolder validates the model keys against the registry and warns if any of them
cannot ship for licence reasons.

## What is shared, and why

| Lives in a pipeline folder | Lives in the shared core |
|---|---|
| which models, which thresholds | the nine stages (`poc/runner/pipeline.py`) |
| your input photographs | the model adapters (`poc/models/`) |
| that pipeline's results | drawing and tables (`poc/runner/report.py`) |
| the two entry points | the orchestrator (`poc/runner/run_combination.py`) |

**The logic is deliberately not copied per pipeline**, and this is the one design
decision worth defending. Copying it would make each folder fully self-contained and
would wreck the codebase: fourteen copies drift, and then nobody can tell which is
correct.

That is not hypothetical here. `poc/pipeline.ipynb` carries its own inline copy of the
stage logic, has never been executed, and now silently disagrees with the CLI. And the
coffee-table volume bug — a 19% over-estimate — survived partly because the notebook
had its own copy of the vocabulary lookup. Both are arguments for one copy, imported.

So a pipeline folder answers *what settings, what input, what output*; the core
answers *how*. Fix a stage once and every pipeline gets the fix.

The notebooks are **generated** from `make_notebook.py` for the same reason: the
narration is identical across pipelines and differs only in which models it names, so
one source regenerates them all. Edit `make_notebook.py`, not a notebook, then
`--all`.

## A limitation to know about

One run processes **one room**. `data/input/` may hold several rooms, but each needs
its own invocation, and each pays the detector's ~370 s warm-up again. Ten rooms is
over an hour of warm-up alone.

For threshold work use the notebook, which keeps the kernel alive and costs ~3 s per
setting. For detection scoring across many images use
`python -m poc.runner.run_dataset_eval --limit N`, which loads the model once.

Batching several rooms into a single invocation is not implemented.
