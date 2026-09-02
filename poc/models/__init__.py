"""
FILE PURPOSE
    Marks poc/models as a package and re-exports the two things callers actually need:
    the data contracts (from .base) and the registry.

WHY THE PACKAGE EXISTS
    So detector / depth / classifier choices become interchangeable strings in a config
    instead of hard-coded imports. That is the whole basis of the comparison harness.

WHAT'S IN HERE
    base.py                 the contracts every adapter satisfies
    registry.py             key -> adapter, plus licence and availability metadata
    detect_*.py             object detectors
    depth_*.py              metric depth estimators
    segment_*.py            box -> mask refinement
    classify_*.py           Method A classifiers

    Run `python -m poc.models.registry` to print the catalogue.
"""
from .base import (Detection, DepthResult, ClassifiedItem, ModelInfo,
                   Detector, DepthEstimator, Classifier, Segmenter,
                   pick_device, nms)

# `registry` is deliberately NOT imported here. Importing it eagerly makes
# `python -m poc.models.registry` emit a RuntimeWarning about the module already
# being in sys.modules. Submodule access still works: `from poc.models import registry`.
__all__ = ["Detection", "DepthResult", "ClassifiedItem", "ModelInfo",
           "Detector", "DepthEstimator", "Classifier", "Segmenter",
           "pick_device", "nms", "registry"]


def __getattr__(name):
    # Must use importlib, NOT `from . import registry`. The latter re-enters this
    # __getattr__ via the import system's hasattr probe and recurses forever.
    if name == "registry":
        import importlib
        return importlib.import_module(".registry", __name__)
    raise AttributeError(name)
