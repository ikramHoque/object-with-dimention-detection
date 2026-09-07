# Running the POC

> **FILE PURPOSE** — Every way to run this, once. Consolidated from the former
> `RUNBOOK.md`, `RUN-LOCAL.md` and `COLAB.md`, which said setup three times,
> the environment variables three times, and "use your own footage" twice.
>
> The pipeline: `PIPELINE.md` · The models: `MODELS.md` · Open decisions: `GAPS.md`

---

## 1 · Setup, once

```bash
./poc/setup.sh                    # ~20 min, ~2.5 GB, mostly torch
source poc/.venv/bin/activate
```

It creates `poc/.venv` on Python 3.12, installs everything, registers the **NX
Survey POC** Jupyter kernel, and verifies all 11 imports so a broken environment
fails in two seconds rather than 20 minutes into a run. It is idempotent — re-run
it after a failure. `NX_RECREATE_VENV=1` forces a rebuild.

### The two environment variables that matter

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1    # always, on a Mac. Prevents hard crashes.
export NX_DEVICE=cpu                    # only when MPS fails outright.
```

`PYTORCH_ENABLE_MPS_FALLBACK` is not optional: some operators have no Metal kernel
and will abort the process instead of falling back. `NX_DEVICE` accepts `cpu`, `mps`
or `cuda` and is respected by every entry point.

---

## 2 · Run a pipeline — the normal case

Each pipeline owns its config, both entry points, its input and its output:

```bash
cp my_room.jpg poc/pipelines/grounding_dino__moge2/data/input/

python -m poc.pipelines.grounding_dino__moge2.run                 # one shot
jupyter lab poc/pipelines/grounding_dino__moge2/notebook.ipynb    # stage by stage
```

Both read that folder's `config.py`, so they cannot disagree — verified: the CLI and
the notebook produce an identical inventory total on the same photograph.

`--input` may be omitted when `data/input/` holds exactly one thing. With several it
lists them and stops rather than guessing.

**Output**, in that pipeline's `results/`:

| file | what it is |
|---|---|
| `<run>.json` | the full record, for `poc.runner.compare` |
| `<run>/frame_000.jpg` | each object boxed and labelled with class, W×D×H and m³ |
| `<run>/items.csv` | per-object measurements and the inventory, for Excel |

Colour in the annotated frame is provenance, not decoration: **green** measured,
**amber** depth assumed from the cube table, **red** detected but unnamed, **blue**
the door used as the scale reference. A frame full of amber and red means the volume
rests on assumptions — and that view is what exposed a 19% over-estimate which looked
perfectly plausible in the JSON.

### Your own input, three shapes

```bash
# one photo
cp ~/Desktop/room.jpg poc/pipelines/grounding_dino__moge2/data/input/

# one room, several angles — the real product shape
mkdir -p poc/pipelines/grounding_dino__moge2/data/input/smith_bedroom
cp ~/photos/bedroom/*.jpg poc/pipelines/grounding_dino__moge2/data/input/smith_bedroom/

# a video walkthrough
cp ~/videos/walk.mp4 poc/pipelines/grounding_dino__moge2/data/input/
```

**One folder is one ROOM, not one dataset.** Stages 7-8 deduplicate across frames on
the assumption every frame shows the same room; 50 photos of 50 rooms would merge into
a single meaningless inventory.

### Add a pipeline

```bash
python -m poc.pipelines.scaffold --detector owlv2 --depth da3_metric
python -m poc.pipelines.make_notebook owlv2__da3_metric__nomask
```

The scaffolder validates model keys against the registry and warns when a combination
cannot ship for licence reasons. See `poc/pipelines/README.md`.

---

## 3 · The other four tools

| Command | Question it answers |
|---|---|
| `python -m poc.models.registry` | which models are installed, and which can ship |
| `python -m poc.runner.run_combination --input <f>` | a one-off run or ablation outside any pipeline folder |
| `python -m poc.runner.run_dataset_eval --limit 100` | does the detector find boxes? Scored against HomeObjects-3K |
| `python -m poc.runner.compare` | rank finished runs by bias |

**There is no run-everything command, on purpose.** All 14 combinations are pipeline
folders, each run explicitly. A single command that fires 14 runs at 6 minutes of
warm-up each hides two hours of work behind one keystroke and produces results nobody
inspects; running one and reading its annotated frame is how the last three real bugs
were found.

### Detection scoring, no footage needed

```bash
python -m poc.datasets.homeobjects --download            # ~390 MB, once
python -m poc.runner.run_dataset_eval --limit 100
python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30
```

This scores **detection only** — it never computes a dimension or a volume, because
the dataset has no volume labels. It tells you whether the pipeline can *see*, never
whether it can *count*.

---

## 4 · Speed — measured, not estimated

M1, macOS 26.4, torch 2.14.0, MPS, 3 Sept 2026:

| Task | Time |
|---|---|
| `setup.sh` | ~20 min, once |
| First model download (DINO + MoGe ≈ 2 GB) | ~10 min, once |
| **Grounding DINO, first inference in a process** | **~370 s** ⚠️ |
| Grounding DINO, every later image | 2.7 s |
| MoGe-2, first inference in a process | ~11 s |
| MoGe-2, every later image | 2.9 s |
| **Detector + depth, steady state** | **~5.7 s / image** |
| 50 dataset images | ~8–10 min |
| Full 14-combination sweep, one room | 2–3 hours |

### The warm-up dominates short runs — batch accordingly

That 370 s is with weights **already cached**. It is a one-time cost per *process*,
and the cause is not yet established — most likely MPS kernel compilation or a silent
CPU fallback on one operator.

```
50 images in ONE process   ->  370 + 50×2.7  ≈  8.5 min
50 images, ONE AT A TIME   ->  50 × 373       ≈  5.2 HOURS
```

So **never loop the CLI over images from a shell script.** Use `--limit`/`--max-frames`
and one invocation. For threshold work use the notebook, which keeps the kernel alive
and costs ~3 s per setting instead of ~6 minutes. A 14-combination sweep pays the
warm-up 14 times, which is most of its 2–3 hours.

---

## 5 · Troubleshooting


| Symptom | Cause | Fix |
|---------|-------|-----|
| `NotImplementedError: ... MPS backend` | op has no Metal kernel | `export PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Still crashes on MPS | op hard-fails even with fallback | `export NX_DEVICE=cpu` for that run |
| 15 s+ per image | silently running on CPU | check `mps available: True`; confirm the run prints `on mps` |
| `MPS backend out of memory` | 16 GB shared with the OS | use `moge-2-vits-normal`, drop `resize_long_edge` to 768, lower `--limit` |
| Kernel dies with no message | memory | same as above; close other apps |
| `401` / `gated repo` on SAM 3 | weights need acceptance | accept the licence on the model page, then `huggingface-cli login` |
| `No solution found ... torch>=2.9.0+cu130 has no wheels with a matching platform tag` | you installed MoGe from `main`, which is MoGe-3 and CUDA-only | already fixed — `setup.sh` pins MoGe to v2.0.0. If you hit this, you are running an old `setup.sh`; `git pull` and re-run. See below. |
| `No module named 'moge'` | the MoGe step of setup failed | `uv pip install --python poc/.venv/bin/python --no-deps git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec` |
| `No module named 'utils3d'` | MoGe was installed `--no-deps` but utils3d was skipped | `uv pip install --python poc/.venv/bin/python -r poc/requirements.txt` |
| `post_process_grounded_object_detection() got an unexpected keyword` | transformers version drift | already handled by a fallback; if it still fails, pin `transformers==4.44` |
| Slow first run only | downloading weights | normal, cached afterwards |
| `ANTHROPIC_API_KEY` missing | Method A needs it | export it, or use `--classifier none` for geometry only |

### Why MoGe is pinned (and why you must not un-pin it)

`setup.sh` installs MoGe like this, and the two flags are both load-bearing:

```bash
uv pip install --no-deps git+https://github.com/microsoft/MoGe.git@b942f00
```

MoGe's `main` branch is no longer MoGe-2. It is **MoGe-3**, and its packaging
declares:

```toml
[tool.uv]         environments = ["sys_platform == 'linux'", "sys_platform == 'win32'"]
[tool.uv.sources] torch = [{ index = "pytorch-cu130" }]
```

macOS is not in that list, and uv honours the CUDA index, so resolution fails
outright:

```
× No solution found when resolving dependencies:
╰─▶ Because torch>=2.9.0+cu130 has no wheels with a matching platform tag
    (e.g., `macosx_26_0_arm64`) ...
```

MoGe-3 also depends on **flex-gemm**, a CUDA GEMM extension with no Apple
silicon build, and on `opencv-python`, which collides with the
`opencv-python-headless` we already install — both provide `cv2`.

The pin costs us nothing. `poc/models/depth_moge2.py` imports
`moge.model.v2` and nothing else, and that file's only third-party imports are
torch, `utils3d` and `huggingface_hub`. `--no-deps` skips gradio, click and
trimesh, which inference never touches; `utils3d` and `scipy`, which it does
need, are pinned in `requirements.txt` instead.

**If you want MoGe-3 later**, that is a Colab/Linux decision, not a local one —
and it is a new model with new published numbers, so it belongs in a fresh
combination, not a silent swap under the existing one.
---

## 6 · Which file do I edit to change…

| …this | edit |
|---|---|
| which models a pipeline uses | `poc/pipelines/<name>/config.py` |
| a detector threshold for one pipeline | same `config.py` (`BOX_TH`, `TEXT_TH`) |
| the prompt list | `poc/detect_vocab.json` |
| a size class or its volume | `poc/cube_table.json` — **placeholder, not NX's sheet** |
| the measurement maths | `poc/runner/pipeline.py` — shared by every pipeline |
| the annotated frame or the CSV | `poc/runner/report.py` — shared by CLI and notebook |
| the notebook narration | `poc/pipelines/make_notebook.py`, then `--all` |
| ground truth | `poc/ground_truth.csv` (copy the template) |

Never edit a `notebook.ipynb` directly — it is generated, and the next `--all` would
overwrite you.

---

## 7 · When to move off the Mac

The M1 is adequate: about 10 minutes for 50 images. Move to a GPU only when one of
these is *actually* true, not in anticipation:

- you need the full 14-combination sweep repeatedly, not once
- you are fine-tuning, not doing inference
- an operator has no Metal kernel and `NX_DEVICE=cpu` is unbearably slow

The real M1 risk is **missing operations**, not speed — hence
`PYTORCH_ENABLE_MPS_FALLBACK=1`. While iterating, the smaller depth variants
(`moge-2-vits-normal`) cut memory and time substantially.

**VSCode cannot officially attach to a Colab runtime.** If you want VSCode plus a GPU,
rent one and use Remote-SSH.


### Option A — Colab in the browser (free T4)

Simplest thing that works today. Notebook provided: `poc/colab/baseline_colab.ipynb`.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. **File → Upload notebook** → pick `poc/colab/baseline_colab.ipynb`
   *(naming: `poc/colab/<pipeline_name>_colab.ipynb`)*
4. Run the cells top to bottom

It clones the repo, installs dependencies, downloads the dataset and runs the evaluation.
Results are written into the Colab session, and the last cell zips them so you can download
and commit them locally.

**Trade-off:** you edit in a browser, not VSCode.

### Option B — Kaggle Notebooks (free T4 or P100, ~30 h/week)

Same browser trade-off, more generous quota, sessions up to 12 h. Better than Colab if you
are running long sweeps. Notebooks can be pushed from the CLI via the Kaggle API, which
gets you closer to a real workflow.

### Option C — Rent a GPU, use VSCode Remote-SSH ← *the real answer if you want VSCode*

Providers like RunPod, Vast.ai or Lambda give you **actual SSH access** for roughly
**$0.20–0.40/hour** for a T4 or A10. Because SSH is supported rather than smuggled:

```bash
# 1. start a GPU pod, note its ssh host/port
# 2. in VSCode: Cmd+Shift+P -> "Remote-SSH: Connect to Host"
# 3. open the folder, pick the interpreter — everything works normally
```

You get real VSCode: debugger, extensions, integrated terminal, and the repo checked out on
a GPU box. **This is the correct setup for sustained work.** A full sweep costs cents.

### Option D — VSCode Jupyter extension → remote Jupyter server

If you already have a GPU box, run a Jupyter server on it and connect natively:

```bash
# on the GPU box
pip install jupyter
jupyter server --no-browser --port 8888 --IdentityProvider.token='pick-a-token'
```

Then in VSCode: open the notebook → click the kernel picker top-right → **Select Another
Kernel → Existing Jupyter Server** → paste `http://<host>:8888/?token=pick-a-token`.

Officially supported, no tunnel hacks. Works with Option C, not with Colab.

---