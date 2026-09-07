# Pipeline · det_rtdetr

How much do we lose by giving up open vocabulary for speed and a clean licence?

| | |
|---|---|
| detector | `rtdetr` |
| depth | `moge2` |
| segmenter | `none` |
| classifier | `claude_sonnet5` |
| thresholds | box `0.3`, text `0.25` |

## Run it

**Put input here:** `data/input/` — a photo, a video, or **a folder of stills of one
room** (one folder is one room, not one dataset).

```bash
# CLI, one shot
python -m poc.pipelines.det_rtdetr.run --input my_room.jpg

# the same pipeline, one stage per cell
jupyter lab poc/pipelines/det_rtdetr/notebook.ipynb
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
