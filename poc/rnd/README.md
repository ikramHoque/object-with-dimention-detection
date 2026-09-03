# R&D — research ceiling measurements

> **FILE PURPOSE** — A quarantined place to run the models we are **not allowed to ship**,
> so we can find out how much accuracy staying licence-clean actually costs us.

---

## Read this first

Everything run from this directory uses models that **cannot go into the NX product**:

| Model | Licence | Why it cannot ship |
|-------|---------|--------------------|
| `yolo_world` | **AGPL-3.0** (Ultralytics) | Would oblige NX to publish the entire source code of the survey tool, publicly. Escapable only by buying an Ultralytics Enterprise Licence |
| `unidepth2` | **CC BY-NC 4.0** | Non-commercial use only. No paid escape exists |

**Ultralytics' published position covers internal R&D**, commercial or not. So even the
runs in this directory may need clearance. **Get NX's legal view before running the
YOLO-World combinations.** The UniDepthV2 ones are research use of a research-licensed
model, which is the case CC BY-NC is written for — but confirm that too.

## Why run them at all

To answer one question with a number instead of a guess:

> **How much accuracy are we giving up by only using models we can ship?**

If the licence-clean stack lands within a point or two of the research ceiling, the licence
costs us nothing and the question is closed forever. If the gap is large, that is a
concrete input to a real decision — buy the Ultralytics Enterprise Licence, negotiate with
ETH Zürich over UniDepth, or invest in hardware depth instead.

Either answer is worth having. Neither is worth accidentally shipping.

## Quarantine

- Results are written to `poc/rnd/results/`, **not** `poc/results/`, so a research number
  can never appear in a production comparison table by accident.
- Every result carries `"shippable": false` and `"rnd": true`.
- `poc/runner/compare.py` reads only `poc/results/`. It will never see these.

## Running it

```bash
# see what it would do, and the licence warnings, without running anything
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 --dry-run

# actually run it (requires the extra installs below)
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 \
    --i-have-legal-clearance
```

The clearance flag is deliberately awkward to type. It is a speed bump, not security.

### Extra installs

Neither model is installed by default. Uncomment in `poc/requirements.txt`:

```
# UniDepthV2 — CC BY-NC
git+https://github.com/lpiccinelli-eth/UniDepth.git

# YOLO-World — AGPL-3.0 via Ultralytics
ultralytics>=8.3
```

## What it reports

A side-by-side of the best **shippable** configuration against the best **research**
configuration, and the gap between them:

```
LICENCE-CLEAN CEILING
  best shippable      grounding_dino + moge2      bias  -4.2%
  best research       grounding_dino + unidepth2  bias  +1.2%
  cost of staying clean                                 3.0 pts
```

Read that number, then decide. Do not ship the right-hand column.
