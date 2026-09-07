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
    DIAGRAM.md                      the system design, for explaining it
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


## The 14 combinations

**The folder name is the recipe**, read left to right:

```
grounding_dino__moge2__sam2__sonnet5
└─ detector ──┘ └depth┘ └seg┘ └classifier┘

NOSHIP_AGPL__yolo_world__moge2__sonnet5
└── cannot ship ──┘
```

Every run is the baseline with **exactly one thing changed**, so a difference in the
result has exactly one possible cause. The full cartesian product would be 288 runs and
about 19 hours, and most cells answer no question anyone asked.

### Shippable (11)

| pipeline folder | the question it answers |
|---|---|
| `grounding_dino__da3_metric__sonnet5` | Does Depth Anything 3's metric variant beat MoGe-2 on our footage? |
| `grounding_dino__moge2__haiku45` | Is Haiku 4.5 good enough at a fifth of the cost? |
| `grounding_dino__moge2__nocls` | How does pure geometry do with no language model at all? |
| `grounding_dino__moge2__opus5` | Is Opus 5 worth 2.5x the cost for size-class accuracy? |
| `grounding_dino__moge2__sam2__sonnet5` | Does bolting SAM 2 masks onto a box detector match a native mask model? |
| `grounding_dino__moge2__sonnet5` | What does the licence-clean default configuration achieve? |
| `grounding_dino__moge2__sonnet5__noanchor` | THE KEY EXPERIMENT. How much does the door anchor actually buy? |
| `owlv2__moge2__sonnet5` | Does OWLv2's long-tail training beat Grounding DINO on unusual furniture? |
| `rtdetr__moge2__sonnet5` | How much do we lose by giving up open vocabulary for speed and a clean licence? |
| `sam3__moge2__sam2__sonnet5` | Everything we expect to be best, together. Run this LAST, after the single-factor runs point at it |
| `sam3__moge2__sonnet5` | Do SAM 3's native masks improve Method B over boxes alone? |

### Cannot ship — evaluation only (3)

Marked three ways: the `NOSHIP_` folder prefix, an `ALERT_DO_NOT_SHIP.md` inside, and
`SHIPPABLE = False` in `config.py`. Their purpose is to measure **what staying
licence-clean costs us**: if one scores materially better than a shippable pipeline,
that is an argument for buying a licence, not permission to ship. Every run needs
`--allow-noncommercial` and results are stamped `shippable: false`.

| pipeline folder | the question it answers |
|---|---|
| `NOSHIP_AGPL__yolo_world__moge2__sonnet5` | Is YOLO-World's 20x speed advantage free, or does accuracy drop? AGPL - comparison only |
| `NOSHIP_AGPL__yoloe__moge2__sonnet5` | Does YOLOE deliver real-time open vocabulary AND masks? AGPL - comparison only |
| `NOSHIP_NC__grounding_dino__unidepth2__sonnet5` | CEILING CHECK. How much accuracy does staying licence-clean cost us? CC BY-NC - cannot ship |

**Read the ALERT file before running these.** Ultralytics' published position is that
any use of their models — internal research included — requires either releasing your
whole project under AGPL-3.0 or buying an Enterprise Licence. That is a legal question,
not a technical one.

### There is no run-everything command, deliberately

Fourteen runs at ~6 minutes of warm-up each hides two hours behind one keystroke and
produces a pile of results nobody inspects. Every real bug so far — the coffee-table
19% over-estimate, the door-anchor percentile error, the unnamed-duplicate double
count — was found by running **one** pipeline and looking at its annotated frame.

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

That is not hypothetical here. `poc/pipeline.ipynb` carried its own inline copy of the
stage logic, was never executed, and had drifted into disagreeing with the CLI — it has
since been deleted for that reason. The coffee-table volume bug, a 19% over-estimate,
also survived partly because a notebook kept its own copy of the vocabulary lookup.
Both are arguments for one copy, imported.

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
