# Running on a GPU — Mac vs Colab vs rented

> **FILE PURPOSE** — Where to actually run this, if not locally. Read §1 first: the honest
> answer is usually that your Mac is fine.
>
> **For step-by-step local instructions see `RUN-LOCAL.md`.** This file is only about the
> GPU alternatives and when they're worth the setup.

---

## 1 · Your M1 is probably fast enough. Don't add cloud complexity yet.

Rough per-image cost on an M1 16 GB using PyTorch's `mps` backend:

| Workload | Estimated time |
|----------|----------------|
| Grounding DINO, one image | ~1.5–4 s |
| MoGe-2 ViT-L, one image | ~1.5–4 s |
| **50 dataset images, detection only** | **~2–4 minutes** |
| One room, 16 frames, detector + depth | ~1–2 minutes |
| Full 14-combination sweep, one room | ~1–2 hours |

For the first attempt — DINO + depth on 50 HomeObjects images with manual review — that is
**minutes**. Setting up a GPU first would cost more time than it saves.

### The real M1 risk is not speed, it's missing operations

Some PyTorch ops have no Metal kernel. When one is hit you get a hard error rather than
slowness. Fix it before you start:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

That silently runs unsupported ops on the CPU instead of crashing. Slower for those ops,
but it works. **Set this in every shell where you run the pipeline.**

If a model still fails on `mps`, force CPU for that run rather than debugging Metal:

```python
CFG["force_device"] = "cpu"     # or: DEVICE = torch.device("cpu")
```

### Use the small model variants while iterating

```python
depth_model = "Ruicheng/moge-2-vits-normal"   # instead of moge-2-vitl
```

Roughly 4× faster, slightly worse. Perfect for "does the plumbing work", wrong for final
numbers.

**Move to a GPU when** you're running the full sweep across several rooms repeatedly, or
when you want YOLO26 comparisons at their real speed. Not before.

---

## 2 · Can VSCode connect to a Colab GPU? Not officially.

Straight answer: **no.** Google does not support attaching an external IDE to a Colab
runtime.

What people do is run a Jupyter server *inside* Colab and expose it through an
`ngrok`/`cloudflared` tunnel, then point VSCode's Jupyter extension at that URL. It
technically works, and I would not build on it:

- Colab's terms discourage using it as a general remote shell or tunnel endpoint; accounts
  have been restricted for it.
- The free tier disconnects on ~90 minutes idle and caps around 12 hours.
- The tunnel drops and takes your kernel state with it.
- Every session needs re-setup.

Fine for a one-off experiment. Not a place to keep a POC.

---

## 3 · What actually works, ranked

### Option A — Colab in the browser (free T4)

Simplest thing that works today. Notebook provided: `poc/colab/baseline_colab.ipynb`.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. **File → Upload notebook** → pick `poc/colab/baseline_colab.ipynb`
   *(naming: `poc/colab/<combination_id>_colab.ipynb`, ids from `combinations.json`)*
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

## 4 · My recommendation for your next step

```
  1. Run the first attempt LOCALLY.       export PYTORCH_ENABLE_MPS_FALLBACK=1
     50 images, DINO + MoGe-2.            ~4 minutes. It will probably just work.

  2. Hit a wall?  ->  which wall?
       too slow          -> Option C (rent a GPU, VSCode Remote-SSH)
       MPS op missing    -> force CPU for that model, or Option A for one run
       want YOLO26 too   -> Option A or C (and see the licence note below)

  3. Only set up cloud when step 1 has actually failed.
```

---

## 5 · A licence note on the YOLO26 validation plan

YOLO26 and YOLO26-depth are **Ultralytics, therefore AGPL-3.0**, exactly like YOLO-World.
So is HomeObjects-3K.

Using them to *check* our numbers is benchmarking, which is what `poc/rnd/` exists for.
Using them *in the product* is the thing that would oblige NX to publish all its source
code. Keep that boundary visible:

```bash
# benchmark — quarantined, tagged shippable=false
python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --i-have-legal-clearance
```

One more thing worth being clear about: **YOLO26-depth agreeing with MoGe-2 does not mean
either is right.** Two models agreeing is consistency, not correctness. For volume you
still need a tape measure — see gap **A1**. What the comparison *does* tell you is whether
MoGe-2 is an outlier, which is worth knowing.
