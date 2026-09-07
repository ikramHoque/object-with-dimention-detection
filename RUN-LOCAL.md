# Running on your M1 Mac

> **FILE PURPOSE** — Exact commands to get the pipeline running locally, what to expect at
> each step, and how to fix the things that actually go wrong on Apple Silicon.
>
> GPU alternatives: `COLAB.md` · Learning the models: `LEARN.md`

**Short version: your M1 is fast enough for this.** 50 dataset images with Grounding DINO
is under 10 minutes. Don't set up cloud until something actually fails.

---

## Step 0 · One-time setup (~20 min, ~2.5 GB)

Your system Python is 3.9, which current PyTorch will not run on, so this builds an
isolated 3.12 environment. Nothing else on your Mac is touched.

```bash
cd /Users/ikramul/Workspace/Office-Workspace/RND/Nippon
./poc/setup.sh
```

It installs `uv` if missing, creates `poc/.venv` on Python 3.12, installs everything in
`requirements.txt`, then MoGe-2 from source (pinned — see below), registers a
Jupyter kernel, and finally verifies that every import works.

```bash
source poc/.venv/bin/activate
```

### Set this before every run — it is not optional

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Some PyTorch operations have no Metal (Apple GPU) kernel. Without this you get a hard
crash; with it, those specific operations quietly run on the CPU instead. Slower for them,
but it works.

Add it to your shell profile so you stop forgetting:

```bash
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
```

### Check what you've got

```bash
python -c "
import torch
print('torch', torch.__version__)
print('mps available:', torch.backends.mps.is_available())
print('cuda available:', torch.cuda.is_available())
"
```

Expect `mps available: True`, `cuda available: False`. That's correct on a Mac — MPS *is*
your GPU.

```bash
python -m poc.models.registry
```

Shows all 13 models and which are installed. Everything except the AGPL and non-commercial
ones should say `yes`.

---

## Step 1 · Get the dataset (~390 MB, once)

```bash
python -m poc.datasets.homeobjects --download
```

Then it prints a summary. **The number to look at is `images_with_a_door`** — that's how
many validation images can exercise the scale anchor at all.

---

## Step 2 · Your first real number (~8–10 min, most of it one-time warm-up)

```bash
python -m poc.runner.run_dataset_eval --detector grounding_dino --limit 50
```

First run also downloads the Grounding DINO weights (~700 MB), so allow a few extra
minutes once.

**What you'll see:**

```
  precision 0.xxx   recall 0.xxx   F1 0.xxx
  tp NN  fp NN  fn NN  (ground truth NN)
  detections we cannot score on this dataset: NN

  class            recall     prec    gt
  chair             0.xxx    0.xxx    NN
  door              0.xxx    0.xxx    NN   <- SCALE ANCHOR: low recall here breaks stage 4
  wardrobe          0.xxx    0.xxx    NN
```

**Read it in this order:**

1. **`door` recall.** Under ~0.6 and stage 4 has nothing to anchor on. That's a finding, not
   a failure — it means the anchor needs the customer-confirmation fallback (gap A5 option c).
2. **`wardrobe` recall.** Core removal item, hardest class here.
3. **Overall recall** before precision. A missed wardrobe is 100% error on that item; a
   false positive costs the reviewer two seconds.

### Sanity-check the speed

Note `sec_per_image` in the output. On an M1 expect **2.5–3 s** once warm — but the
first image of a run also carries a ~370 s one-time warm-up, so a 1-image run looks
catastrophically slow and a 50-image run does not. If later images are 15 s+,
something fell back to CPU — see troubleshooting.

---

## Step 3 · See the parameter trade-off yourself (~10–20 min)

This is Lesson 1 made concrete. Same images, six confidence thresholds:

```bash
python -m poc.runner.run_dataset_eval --compare-thresholds --limit 30
```

```
   box_th  precision   recall      F1     fp     fn
     0.15      0.xxx    0.xxx   0.xxx     NN     NN
     0.20      0.xxx    0.xxx   0.xxx     NN     NN
     ...
```

Watch precision fall as recall rises. **Pick the row you want, not the best F1** — for a
survey, recall wins.

---

## Step 4 · Add depth (diagnostic only)

```bash
python -m poc.runner.run_dataset_eval --detector grounding_dino --depth moge2 --limit 20
```

Downloads MoGe-2 (~1.3 GB) once. It prints a depth sanity check on the first three images:

```
  depth sanity check (DIAGNOSTIC ONLY — nothing here can score it):
    image_001.jpg    range 1.2-7.4m  vertical span 2.4m  valid 94%
    A plausible room: depth 1-8m, vertical span 2-3m, valid >80%.
```

**Why diagnostic only:** HomeObjects-3K has no depth ground truth. You can eyeball whether
the numbers are plausible; you cannot score them. Volume accuracy needs the tape-measure
step — gap **A1**.

**If the vertical span reads like room *width* rather than height**, MoGe-2's axis
convention differs from what stage 4 assumes. Flip `VERT_AXIS` in the notebook, or
`vert_axis` in `depth_moge2.py`.

### Going faster while iterating

MoGe-2 ViT-L is 326M parameters. Use the small variant for plumbing checks:

```python
# poc/models/depth_moge2.py, or in the notebook CFG
depth_model = "Ruicheng/moge-2-vits-normal"   # ~4x faster, slightly worse
```

Right for "does this work". Wrong for final numbers.

---

## Step 5 · Compare detectors on identical images

```bash
for d in grounding_dino owlv2 rtdetr; do
  python -m poc.runner.run_dataset_eval --detector $d --limit 50 --save
done
```

`rtdetr` will have lower recall — it's closed-vocabulary and has no class for 9 of our 26
prompts. That's the point of including it: it measures what open vocabulary is worth.

---

## Step 6 · Your own footage (the part that actually matters)

The dataset validates detection. **Volume needs your own rooms.**

```bash
# 1. film two rooms: slow pan, door visible in frame, 30-60s each
cp ~/Desktop/bedroom.mp4 poc/data/input/

# 2. hand-measure the same rooms — THIS IS THE STEP THAT MAKES IT AN EXPERIMENT
cp poc/ground_truth_template.csv poc/ground_truth.csv
#    then fill it in with a laser measure. Half a day for 3-5 rooms. Gap A1.

# 3. the full room pipeline
export ANTHROPIC_API_KEY=sk-ant-...        # Method A only
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5

# 4. THE KEY EXPERIMENT — same thing, anchor off
python -m poc.runner.run_combination --input bedroom.mp4 --room BED01 \
    --detector grounding_dino --depth moge2 --classifier claude_sonnet5 --no-anchor

# 5. compare
python -m poc.runner.compare --room BED01
```

Expect **1–2 minutes per room** for steps 3 and 4 on an M1.

### Or use the notebook, which shows you the pictures

```bash
jupyter lab poc/pipeline.ipynb
```

Select the **NX Survey POC** kernel. The notebook renders keyframes, detection boxes and
depth maps at each stage, which is far better for judging *why* something went wrong.

---

## Expected timings on an M1 16 GB

**Measured on this machine, 3 Sept 2026** (M1, macOS 26.4, torch 2.14.0, MPS).
The earlier numbers in this table were taken from published benchmarks and were
wrong about the thing that dominates a short run — see the warm-up row.

| Task | Time | How we know |
|------|------|-------------|
| `setup.sh` | ~20 min, once | measured |
| First model download (DINO + MoGe ≈ 2 GB) | ~10 min, once | measured |
| **Grounding DINO, first inference in a process** | **~370 s** ⚠️ | measured, weights already cached |
| Grounding DINO, every later image | **2.7 s** | measured |
| MoGe-2 ViT-L, first inference in a process | ~11 s | measured |
| MoGe-2 ViT-L, every later image | **2.9 s** | measured |
| Detector + depth, steady state | **~5.7 s / image** | measured |
| 50 dataset images, detection | **~8–10 min** | 370 s warm-up + 50 × 2.7 s |
| One room, 16 frames, detector + depth | ~8 min | 370 s warm-up + 16 × 5.7 s |
| Full 14-combination sweep, one room | 2–3 hours | warm-up is paid per combination |

### The warm-up dominates short runs — batch accordingly

Grounding DINO's **first** inference in a fresh process costs about **370 seconds**,
with the weights already on disk. Every subsequent image costs 2.7 s. We have not
yet established the cause; the likely candidates are MPS kernel compilation or a
silent CPU fallback on one operator.

The practical consequence is what matters:

```
50 images in ONE process   ->  370 + 50×2.7  ≈  8.5 min
50 images, ONE AT A TIME   ->  50 × 373       ≈  5.2 HOURS
```

So **always pass `--limit`/`--max-frames` and process a batch in a single
invocation.** Never loop the CLI over images from a shell script. This also means
a 14-combination sweep pays the warm-up 14 times, which is most of its 2–3 hours.

---

## Running on your own rooms

`--input` takes a single photo, a video, or **a folder of stills of one room** — the
last being the real product shape, since a customer photographs a bedroom from
several angles rather than filming it.

```bash
mkdir -p poc/data/input/smith_bedroom
cp ~/photos/bedroom/*.jpg poc/data/input/smith_bedroom/
python -m poc.runner.run_combination --input smith_bedroom \
    --detector grounding_dino --depth moge2 --classifier none
```

One folder is one **room**, not one dataset: stages 7-8 deduplicate across frames on
the assumption every frame shows the same room. See `PIPELINE.md` section 3.

---

## Troubleshooting

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

### The two environment variables that matter

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1    # always. prevents hard crashes.
export NX_DEVICE=cpu                    # only when MPS fails outright.
```

`NX_DEVICE` accepts `cpu`, `mps` or `cuda` and is respected by both the notebook and every
model adapter.

---

## When to stop using the Mac

Move to a GPU (see `COLAB.md`) when **one of these is actually true**, not before:

- you're running the full sweep across several rooms, repeatedly
- you want YOLO26 / YOLO-World comparisons at their real speed
- an MPS problem is costing more time than a rented GPU would

For the first attempt — detection scored on 50 images, plus depth by eye — local is the
right call.
