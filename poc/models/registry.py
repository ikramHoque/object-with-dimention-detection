"""
FILE PURPOSE
    The catalogue. Maps a short key like "grounding_dino" to the adapter that
    implements it, and knows each model's licence and whether it is installed —
    WITHOUT importing heavy libraries just to answer that question.

WHY IT MATTERS
    Two jobs.
    1. It makes combinations expressible as plain strings, so a run is described by
       config rather than by code. That is what lets us compare many pipelines.
    2. It is the licence gate. Every entry carries commercial_ok, and the runner
       refuses to build a pipeline containing a non-commercial model unless it is
       explicitly overridden. Prototyping on a model that can never ship is a
       specific, expensive mistake this file exists to prevent.

USED BY  poc/runner/*.py
CLI      python -m poc.models.registry        # prints the catalogue + availability

HOW TO RUN
    python -m poc.models.registry
        Prints all 13 models: licence, ship/no-ship, open or fixed vocabulary,
        whether it returns masks, and whether it is installed.
    Run this first when something says "not installed".
"""
from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import asdict

from .base import ModelInfo

# key -> (module, class name). Imported lazily, only when actually requested.
_SPECS: dict[str, tuple[str, str]] = {
    # ---- detectors ----
    "grounding_dino": (".detect_grounding_dino", "GroundingDinoDetector"),
    "owlv2":          (".detect_owlv2",          "Owlv2Detector"),
    "sam3":           (".detect_sam3",           "Sam3Detector"),
    "yolo_world":     (".detect_yolo_world",     "YoloWorldDetector"),
    "yoloe":          (".detect_yoloe",          "YoloeDetector"),
    "rtdetr":         (".detect_rtdetr",         "RtDetrDetector"),
    # ---- depth ----
    "moge2":          (".depth_moge2",           "Moge2Depth"),
    "da3_metric":     (".depth_anything3",       "DepthAnything3Depth"),
    "unidepth2":      (".depth_unidepth",        "UniDepthV2Depth"),
    # ---- segmenters ----
    "sam2":           (".segment_sam2",          "Sam2Segmenter"),
    # ---- classifiers ----
    "claude_sonnet5": (".classify_claude",       "ClaudeClassifier"),
    "claude_opus5":   (".classify_claude",       "ClaudeClassifier"),
    "claude_haiku45": (".classify_claude",       "ClaudeClassifier"),
}

# Constructor kwargs for keys that share a class.
_KWARGS: dict[str, dict] = {
    "claude_sonnet5": {"model": "claude-sonnet-5"},
    "claude_opus5":   {"model": "claude-opus-5"},
    "claude_haiku45": {"model": "claude-haiku-4-5"},
}

# When run as `python -m poc.models.registry`, __name__ becomes "__main__" and
# deriving the package from it breaks every relative import. __package__ stays correct.
_PKG = __package__ or "poc.models"


def info(key: str) -> ModelInfo:
    """Metadata for one key. Imports the adapter module but not its ML backend."""
    if key not in _SPECS:
        raise KeyError(f"unknown model '{key}'. Known: {sorted(_SPECS)}")
    mod_name, cls_name = _SPECS[key]
    mod = importlib.import_module(mod_name, package=_PKG)
    if key in _KWARGS and hasattr(mod, "PRICING"):
        # classifier variants build their own info from the model name
        return getattr(mod, cls_name)(**_KWARGS[key]).info
    return mod.INFO


_BUILT: dict = {}


def build(key: str, *, fresh: bool = False, **overrides):
    """Instantiate an adapter. Still does not load weights — that happens on first use.

    Unsupported keyword arguments are DROPPED WITH A WARNING rather than raising.
    Callers pass a common set of knobs (box_threshold, text_threshold) across a
    catalogue of models that do not all have them — RT-DETR has no text threshold
    because it has no text. Raising would force every caller to special-case each
    model; silently ignoring would let a threshold you thought you set do nothing.
    So it is dropped, loudly.

    WITHIN ONE PROCESS the same (key, kwargs) returns the SAME adapter. Grounding
    DINO's first forward pass costs ~370 s and every one after ~2.7 s, so rebuilding
    per room would make `run.py --all` over 5 rooms a 30-minute job instead of a
    7-minute one. The cache is what makes multi-room runs practical.

    Caveat: because the same object comes back, a threshold you MUTATE on an adapter
    stays mutated for the next caller asking for those same kwargs. Code that
    temporarily changes `det.box_th` must restore it — the notebook sweep cells do.
    Pass fresh=True for a genuinely new instance.
    """
    import inspect
    mod_name, cls_name = _SPECS[key]
    mod = importlib.import_module(mod_name, package=_PKG)
    cls = getattr(mod, cls_name)
    kwargs = {**_KWARGS.get(key, {}), **overrides}
    try:
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):
        accepted = set(kwargs)
    dropped = sorted(k for k in kwargs if k not in accepted)
    used = {k: v for k, v in kwargs.items() if k in accepted}

    # Only the kwargs the class actually accepts identify the instance; a dropped
    # one changes nothing about it, so it must not split the cache.
    try:
        memo = (key, tuple(sorted((k, v) for k, v in used.items())))
    except TypeError:          # an unhashable kwarg — do not cache, just build
        memo = None
    if not fresh and memo is not None and memo in _BUILT:
        return _BUILT[memo]

    for k in dropped:
        print(f"  ! {key} takes no '{k}' — ignoring it (value {kwargs[k]!r})")
    obj = cls(**used)
    if not fresh and memo is not None:
        _BUILT[memo] = obj
    return obj


def _probe_module(extra: str) -> str:
    """pip_extra -> the module name to look for.

    Derived rather than matched against a table of exact URLs. The table broke
    the moment MoGe's URL gained a commit pin (it is now pinned to v2.0.0 —
    poc/requirements.txt explains why): the lookup missed, fell back to the raw
    URL as the module name, and every MoGe combination silently reported
    "not installed".

        git+https://github.com/microsoft/MoGe.git@b942f00     -> moge
        git+https://github.com/lpiccinelli-eth/UniDepth.git   -> unidepth
        transformers                                          -> transformers
    """
    if extra.startswith("git+"):
        repo = extra.split("#")[0].rsplit("/", 1)[-1]   # MoGe.git@b942f00...
        return repo.split(".git")[0].lower()            # moge
    return re.split(r"[<>=!~\[; ]", extra, maxsplit=1)[0].strip()


def is_installed(key: str) -> bool:
    """Is the backend library present? Checks import spec only — no heavy import."""
    probe = _probe_module(info(key).pip_extra or "")
    if not probe:
        return True
    try:
        return importlib.util.find_spec(probe) is not None
    except (ImportError, ValueError):
        return False


def keys(kind: str | None = None) -> list[str]:
    if kind is None:
        return sorted(_SPECS)
    return sorted(k for k in _SPECS if info(k).kind == kind)


def catalogue() -> list[dict]:
    rows = []
    for k in sorted(_SPECS):
        i = info(k)
        rows.append({**asdict(i), "installed": is_installed(k)})
    return rows


def check_licences(selected: list[str], allow_noncommercial: bool = False) -> list[str]:
    """Return the blocking non-commercial models. Empty list means the combo is clean."""
    bad = [k for k in selected if k and not info(k).commercial_ok]
    if bad and not allow_noncommercial:
        return bad
    return []


def _print_catalogue() -> None:
    rows = catalogue()
    order = {"detector": 0, "depth": 1, "segmenter": 2, "classifier": 3}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["key"]))
    kind = None
    print(f"\n{'key':16} {'licence':40} {'ship?':6} {'vocab':6} {'mask':5} inst")
    print("-" * 90)
    for r in rows:
        if r["kind"] != kind:
            kind = r["kind"]
            print(f"\n[{kind.upper()}]")
        print(f"  {r['key']:14} {r['licence']:40} "
              f"{'YES' if r['commercial_ok'] else 'NO ':6} "
              f"{('open' if r['open_vocabulary'] else 'fixed') if r['kind']=='detector' else '-':6} "
              f"{'yes' if r['gives_masks'] else '-':5} "
              f"{'yes' if r['installed'] else 'no'}")
    blocked = [r["key"] for r in rows if not r["commercial_ok"]]
    print(f"\nCANNOT SHIP ({len(blocked)}): {', '.join(blocked)}")
    print("  These are usable for comparison only, and only with --allow-noncommercial.")
    missing = [r["key"] for r in rows if not r["installed"]]
    if missing:
        print(f"\nNOT INSTALLED ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    _print_catalogue()
