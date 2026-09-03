# Running on your M1 Mac

> **FILE PURPOSE** — Exact commands to get the pipeline running locally, what to expect at
> each step, and how to fix the things that actually go wrong on Apple Silicon.
>
> GPU alternatives: `COLAB.md` · Learning the models: `LEARN.md`

**Short version: your M1 is fast enough for this.** 50 dataset images with Grounding DINO
is 2–4 minutes. Don't set up cloud until something actually fails.

---

## Step 0 · One-time setup (~20 min, ~2.5 GB)

Your system Python is 3.9, which current PyTorch will not run on, so this builds an
isolated 3.12 environment. Nothing else on your Mac is touched.

```bash
cd /Users/ikramul/Workspace/Office-Workspace/RND/Nippon
./poc/setup.sh
```

It installs `uv` if missing, creates `poc/.venv` on Python 3.12, installs everything in
`requirements.txt` (including MoGe-2 from source), and registers a Jupyter kernel.

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

## Step 2 · Your first real number (~2–4 min)

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

Note `sec_per_image` in the output. On an M1 expect **1.5–4 s**. If you're seeing 15 s+,
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

| Task | Time |
|------|------|
| `setup.sh` | ~20 min, once |
| First model download (DINO 700 MB + MoGe 1.3 GB) | ~5 min, once |
| Grounding DINO, one image | 1.5–4 s |
| MoGe-2 ViT-L, one image | 1.5–4 s |
| **50 dataset images, detection** | **2–4 min** |
| `--compare-thresholds --limit 30` | 10–20 min |
| One room, 16 frames, detector + depth | 1–2 min |
| Full 14-combination sweep, one room | 1–2 hours |

Only the last row is uncomfortable. Everything before it is fine locally.

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
| `No module named 'moge'` | MoGe install failed | `uv pip install --python poc/.venv/bin/python git+https://github.com/microsoft/MoGe.git` |
| `post_process_grounded_object_detection() got an unexpected keyword` | transformers version drift | already handled by a fallback; if it still fails, pin `transformers==4.44` |
| Slow first run only | downloading weights | normal, cached afterwards |
| `ANTHROPIC_API_KEY` missing | Method A needs it | export it, or use `--classifier none` for geometry only |

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
