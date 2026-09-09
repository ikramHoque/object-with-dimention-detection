"""
FILE PURPOSE
    Creates a new pipeline folder: config, both entry points, data/input, results
    and a README. One command instead of copying a folder and forgetting to edit
    half of it.

HOW TO RUN
    python -m poc.pipelines.scaffold --detector grounding_dino --depth moge2
    python -m poc.pipelines.scaffold --detector owlv2 --depth da3_metric --segmenter sam2
    python -m poc.pipelines.scaffold --detector grounding_dino --depth moge2 --name baseline

    Then generate its notebook:
    python -m poc.pipelines.make_notebook <name>

WHY A GENERATOR AND NOT A TEMPLATE FOLDER
    A template folder gets copied and then edited by hand, and the name inside
    config.py ends up disagreeing with the folder it sits in. Generating from the
    arguments means the folder name, the config and the README cannot disagree.

USED BY  you, when adding a combination
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

CONFIG = '''"""
FILE PURPOSE
    What THIS pipeline is. The single record of which models run, with which
    thresholds, and what question the combination answers.

    Edit this file to change the pipeline. Both entry points read it:
        run.py          the CLI
        notebook.ipynb  the same thing, one stage per cell

    The stage logic is NOT here — it is shared in poc/runner/pipeline.py. See
    poc/pipelines/__init__.py for why that separation is deliberate.
"""

NAME      = "{name}"
QUESTION  = "{question}"
SHIPPABLE = {shippable}        # False -> evaluation only. See ALERT_DO_NOT_SHIP.md

# ---- models -------------------------------------------------------------
DETECTOR  = "{detector}"        # finds and names objects
DEPTH     = "{depth}"        # metres per pixel
SEGMENTER = {segmenter}        # box -> mask. Measures the object, not the wall behind it
CLASSIFIER = {classifier}        # None skips Method A: no API key, no cost

# ---- detector knobs -----------------------------------------------------
# BOX_TH  keep a box at all.   LOWER -> more objects found, more rubbish
# TEXT_TH how sure of the NAME. too HIGH -> unnamed boxes
#                               too LOW  -> prompts fuse, also unnamed
# An object never detected can never be measured, so recall comes first.
BOX_TH    = {box_th}
TEXT_TH   = {text_th}

# ---- run shape ----------------------------------------------------------
MAX_FRAMES = 16          # frames kept from a video or a folder of stills
DEDUP      = "max"       # "max" trusts each sighting; "median" needs agreement
NO_ANCHOR  = False       # True disables the door scale correction (an ablation)
'''

RUN = '''"""
FILE PURPOSE
    CLI entry point for the {name} pipeline.

HOW TO RUN
    python -m poc.pipelines.{name}.run --input my_room.jpg
    python -m poc.pipelines.{name}.run                     # if data/input has one thing
    python -m poc.pipelines.{name}.run --input my_room.jpg --box-th 0.20

    Input comes from   poc/pipelines/{name}/data/input/
    Output goes to     poc/pipelines/{name}/results/

    It is thin on purpose: config.py says what to run, poc/pipelines/_runner.py
    handles the plumbing, and poc/runner/run_combination.py does the work — the
    same code path as the general-purpose CLI, so this cannot drift from it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.pipelines._runner import run_from_config      # noqa: E402
from poc.pipelines.{name} import config                # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_from_config(config))
'''

README = '''# Pipeline · {name}

{question}

| | |
|---|---|
| detector | `{detector}` |
| depth | `{depth}` |
| segmenter | `{segmenter_txt}` |
| classifier | `{classifier_txt}` |
| thresholds | box `{box_th}`, text `{text_th}` |

## Run it

**Put input here:** `data/input/` — a photo, a video, or **a folder of stills of one
room** (one folder is one room, not one dataset).

```bash
# CLI, one shot
python -m poc.pipelines.{name}.run --input my_room.jpg

# the same pipeline, one stage per cell
jupyter lab poc/pipelines/{name}/notebook.ipynb
```

**Output lands in** `results/`:

- `<run>.json` — the full record, for `poc/runner/compare.py`
- `<run>/frame_000.jpg` — boxes labelled with class, W×D×H and m³
- `<run>/items.csv` — per-object measurements and the inventory

Colour in the annotated frame is provenance, not decoration: **green** measured,
**amber** depth assumed from the cube table, **red** detected but unnamed, **blue**
the door used as the scale reference. A frame full of amber and red means the volume
rests on assumptions.

## Changing it

Edit `config.py`. Both entry points read it, so they cannot disagree.

The stage logic is shared, in `poc/runner/pipeline.py` — deliberately not copied
here. `PIPELINE.md` walks all nine stages with the function behind each.

## What this cannot tell you

Whether the numbers are **right**. That needs hand-measured rooms in
`poc/ground_truth.csv` and `--room <id>` to enable stage 9. Until then the output is
what the pipeline believes, not what is true. Gap **A1** in `GAPS.md`.
'''

ALERT = """# DO NOT USE THIS PIPELINE IN PRODUCTION

**{name}**

This pipeline depends on {bad_list}, which **cannot be shipped**:

{bad_table}

## What that means in practice

{why}

## What this folder IS for

Measuring what staying licence-clean costs us. If `{name}` scores materially better
than a shippable pipeline, that number is an argument for buying a licence or for
finding another approach — it is **not** permission to ship this.

Every run requires `--allow-noncommercial`, and the results are stamped
`shippable: false`. That flag is a deliberate speed bump, not a formality.

## Before running this at all

Ultralytics' own published position is that **any** use of their models — including
internal research, commercial or not — requires either releasing your entire project
under AGPL-3.0 or buying an Enterprise Licence. AGPL also reaches SaaS deployment, so
wrapping it in an API is not an escape.

So this is a legal question, not a technical one. **Get BJIT/the client sign-off first.**

See `MODELS.md` for the licence reasoning and `GAPS.md` gap B9.
"""

LICENCE_NOTE = {
    "AGPL-3.0 or paid Ultralytics Enterprise":
        "AGPL-3.0 requires publishing the source of the ENTIRE work that uses it, and\n"
        "it reaches network deployment, so a hosted API does not avoid it. Ultralytics\n"
        "sells an Enterprise Licence as the alternative.",
    "CC BY-NC 4.0":
        "CC BY-NC 4.0 forbids commercial use outright. There is no paid escape hatch —\n"
        "no amount of money makes this shippable in a commercial product.",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create a new pipeline folder.")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--segmenter", default=None)
    ap.add_argument("--classifier", default=None,
                    help="e.g. claude_sonnet5. Omit to skip Method A.")
    ap.add_argument("--box-th", type=float, default=0.30)
    ap.add_argument("--text-th", type=float, default=0.25)
    ap.add_argument("--name", default=None, help="folder name (default: from the models)")
    ap.add_argument("--question", default=None, help="what this combination answers")
    ap.add_argument("--force", action="store_true", help="overwrite config/run/README")
    a = ap.parse_args(argv)

    from poc.models import registry
    for kind, key in (("detector", a.detector), ("depth", a.depth),
                      ("segmenter", a.segmenter), ("classifier", a.classifier)):
        if key and key not in registry.keys(kind):
            print(f"unknown {kind} '{key}'. Available: {', '.join(registry.keys(kind))}")
            return 1

    # Refuse to scaffold something that cannot ship without saying so out loud.
    unshippable = [k for k in (a.detector, a.depth, a.segmenter, a.classifier)
                   if k and not registry.info(k).commercial_ok]
    if unshippable:
        print(f"! {', '.join(unshippable)} cannot ship (licence). This pipeline will")
        print("  need --allow-noncommercial on every run and belongs in poc/rnd/.")

    # The folder name carries the warning. Someone browsing the tree, or importing a
    # module path, should not be able to miss that this one cannot ship.
    prefix = ""
    if unshippable:
        licences = {registry.info(k).licence for k in unshippable}
        prefix = "NOSHIP_AGPL__" if any("AGPL" in l for l in licences) else "NOSHIP_NC__"
    # Match the convention every existing folder already uses:
    #     detector __ depth __ [segmenter] __ classifier
    # e.g. grounding_dino__moge2__sam2__sonnet5, grounding_dino__moge2__nocls.
    # The classifier is the axis we compare with and without, so it has to be in
    # the name; a segmenter is only named when there is one. This used to emit
    # "nomask" and drop the classifier entirely, so an LLM pipeline and its
    # no-LLM twin would have collided on one folder.
    parts = [a.detector, a.depth]
    if a.segmenter:
        parts.append(a.segmenter)
    parts.append(re.sub(r"^claude_", "", a.classifier) if a.classifier else "nocls")
    name = a.name or "__".join(parts)
    if prefix and not name.startswith("NOSHIP"):
        name = prefix + name
    folder = HERE / name
    if folder.exists() and not a.force:
        print(f"{folder.relative_to(HERE.parents[1])} already exists. --force to rewrite "
              f"config.py, run.py and README.md (data/ and results/ are never touched).")
        return 1

    for sub in ("data/input", "results"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()

    question = a.question or f"{a.detector} + {a.depth}" + (
        f" + {a.segmenter}" if a.segmenter else "")
    fmt = dict(name=name, question=question, shippable=not unshippable,
               detector=a.detector, depth=a.depth,
               segmenter=repr(a.segmenter), classifier=repr(a.classifier),
               segmenter_txt=a.segmenter or "none", classifier_txt=a.classifier or "none",
               box_th=a.box_th, text_th=a.text_th)

    (folder / "__init__.py").write_text(f'"""Pipeline · {name}. See config.py."""\n')
    (folder / "config.py").write_text(CONFIG.format(**fmt))
    (folder / "run.py").write_text(RUN.format(**fmt))
    (folder / "README.md").write_text(README.format(**fmt))

    if unshippable:
        rows = "\n".join(f"- **{k}** — {registry.info(k).licence}"
                          for k in unshippable)
        why = "\n\n".join(sorted({LICENCE_NOTE.get(registry.info(k).licence,
                                                     "Not licensed for commercial use.")
                                    for k in unshippable}))
        (folder / "ALERT_DO_NOT_SHIP.md").write_text(
            ALERT.format(name=name, bad_list=", ".join(f"`{k}`" for k in unshippable),
                         bad_table=rows, why=why))
        print(f"  wrote ALERT_DO_NOT_SHIP.md ({', '.join(unshippable)})")

    rel = folder.relative_to(HERE.parents[1])
    print(f"created {rel}/")
    for f in ("config.py", "run.py", "README.md", "data/input/", "results/"):
        print(f"    {f}")
    print(f"\nnext:")
    print(f"    python -m poc.pipelines.make_notebook {name}")
    print(f"    cp your_room.jpg {rel}/data/input/")
    print(f"    python -m poc.pipelines.{name}.run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
