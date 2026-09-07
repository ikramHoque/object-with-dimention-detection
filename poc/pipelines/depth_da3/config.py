"""
FILE PURPOSE
    What THIS pipeline is. The single record of which models run, with which
    thresholds, and what question the combination answers.

    Edit this file to change the pipeline. Both entry points read it:
        run.py          the CLI
        notebook.ipynb  the same thing, one stage per cell

    The stage logic is NOT here — it is shared in poc/runner/pipeline.py. See
    poc/pipelines/__init__.py for why that separation is deliberate.
"""

NAME      = "depth_da3"
QUESTION  = "Does Depth Anything 3's metric variant beat MoGe-2 on our footage?"
SHIPPABLE = True        # False -> evaluation only. See ALERT_DO_NOT_SHIP.md

# ---- models -------------------------------------------------------------
DETECTOR  = "grounding_dino"        # finds and names objects
DEPTH     = "da3_metric"        # metres per pixel
SEGMENTER = None        # box -> mask. Measures the object, not the wall behind it
CLASSIFIER = 'claude_sonnet5'        # None skips Method A: no API key, no cost

# ---- detector knobs -----------------------------------------------------
# BOX_TH  keep a box at all.   LOWER -> more objects found, more rubbish
# TEXT_TH how sure of the NAME. too HIGH -> unnamed boxes
#                               too LOW  -> prompts fuse, also unnamed
# An object never detected can never be measured, so recall comes first.
BOX_TH    = 0.3
TEXT_TH   = 0.25

# ---- run shape ----------------------------------------------------------
MAX_FRAMES = 16          # frames kept from a video or a folder of stills
DEDUP      = "max"       # "max" trusts each sighting; "median" needs agreement
NO_ANCHOR  = False       # True disables the door scale correction (an ablation)
