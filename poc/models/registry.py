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


def build(key: str, **overrides):
    """Instantiate an adapter. Still does not load weights — that happens on first use."""
    mod_name, cls_name = _SPECS[key]
    mod = importlib.import_module(mod_name, package=_PKG)
    kwargs = {**_KWARGS.get(key, {}), **overrides}
    return getattr(mod, cls_name)(**kwargs)


def is_installed(key: str) -> bool:
    """Is the backend library present? Checks import spec only — no heavy import."""
    extra = info(key).pip_extra or ""
    root = extra.split("/")[-1].replace(".git", "") if extra.startswith("git+") else extra
    probe = {"git+https://github.com/microsoft/MoGe.git": "moge",
             "git+https://github.com/lpiccinelli-eth/UniDepth.git": "unidepth"}.get(extra, root)
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
