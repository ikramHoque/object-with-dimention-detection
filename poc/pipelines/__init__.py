"""
FILE PURPOSE
    One folder per pipeline. Each holds the things that belong to THAT pipeline —
    its configuration, its two entry points, its input photographs, its results —
    and imports everything else from the shared core.

WHAT IS IN A PIPELINE FOLDER
    config.py       which models, which thresholds, and the question it answers
    run.py          the CLI:  python -m poc.pipelines.<name>.run --input room.jpg
    notebook.ipynb  the same pipeline, one stage per cell
    data/input/     your photographs for this pipeline
    results/        JSON, annotated frames and items.csv land here
    README.md       what it is and how to run it both ways

WHAT IS DELIBERATELY *NOT* IN A PIPELINE FOLDER
    The pipeline logic. Stage maths lives in poc/runner/pipeline.py, models in
    poc/models/, drawing and tables in poc/runner/report.py — one copy, shared.

    This is the whole design decision, so it is worth stating plainly: copying the
    logic into each folder would make every folder self-contained and would destroy
    the codebase. Fourteen copies drift, and then no one can tell which is correct.
    We already have proof: poc/pipeline.ipynb carries its own inline copy of the
    stage logic and now silently disagrees with the CLI, and the coffee-table
    volume bug survived partly because the notebook had its own vocabulary lookup.

    So a pipeline folder answers "what settings, what input, what output" and the
    core answers "how". Change a stage once and every pipeline gets it.

ADDING A PIPELINE
    python -m poc.pipelines.scaffold --detector owlv2 --depth da3_metric
    which writes the folder, a config, both entry points and a README.
"""
