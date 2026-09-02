"""
FILE PURPOSE
    Classifier adapter · Claude vision. This IS Method A.

WHAT IT DOES
    Looks at a frame and names each object's SIZE CLASS — not its dimensions. The
    class carries the volume, so "sofa_3_seat" is a complete answer.

WHY CLASSIFICATION BEATS MEASUREMENT HERE
    Both shipping products in this market work this way, and they advertise exactly
    this distinction: "queen mattress vs king mattress", "two-seater vs sectional".
    They are not measuring those. Choosing between 3 sofa classes is a far easier
    problem than getting 3 continuous numbers right.

HOW THE OUTPUT IS FORCED
    A tool schema whose `enum` is the cube table. The model physically cannot return
    a class we have no volume for, so no output validation is needed.

WHAT IT IS NOT USED FOR
    Counting. Measured VLM counting accuracy is around 0.53 and it UNDER-counts,
    which produces confident low inventories and undersized trucks. The detector
    counts; this names.

MODELS  Sonnet 5 is the default. Opus 5 for hard rooms, Haiku 4.5 for cheap triage —
        comparing the three is one axis of the sweep.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import ClassifiedItem, ModelInfo

# $ per million tokens (input, output), for the cost line in results.
PRICING = {
    "claude-opus-5":      (5.0, 25.0),
    "claude-sonnet-5":    (2.0, 10.0),
    "claude-haiku-4-5":   (1.0,  5.0),
}

INFO = ModelInfo(
    key="claude_sonnet5",
    kind="classifier",
    display="Claude Sonnet 5 (vision)",
    checkpoint="claude-sonnet-5",
    licence="commercial API",
    commercial_ok=True,
    pip_extra="anthropic",
    notes="~$0.02/frame. Forced tool output guarantees a valid size class.",
)

PROMPT = """You are cataloguing a room for a household removal survey.

List every MOVABLE item you can see. For each one pick the single closest size_class from
the allowed list — the class carries the packed volume, so choosing between e.g.
sofa_2_seat / sofa_3_seat / sofa_sectional matters as much as recognising it is a sofa.

Rules:
- Count only what is visible IN THIS IMAGE. Do not infer items you cannot see.
- Do not count fitted or structural things: fitted kitchen units, built-in wardrobes,
  radiators, doors, windows, flooring, light fittings.
- If an item is partly hidden, still count it once and say so in reasoning.
- If unsure between two size classes, pick the smaller and lower your confidence.
- confidence is 0.0-1.0 for the size_class choice, not for the item existing."""


class ClaudeClassifier:
    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self.info = ModelInfo(
            key=f"claude_{model.replace('claude-', '').replace('-', '')}",
            kind="classifier",
            display=f"Claude vision ({model})",
            checkpoint=model,
            licence="commercial API",
            commercial_ok=True,
            pip_extra="anthropic",
            notes=INFO.notes,
        )
        self._client = None

    def _c(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()
        return self._client

    def classify(self, image_bgr: np.ndarray, allowed: Sequence[str]):
        """Returns (items, usage). usage carries token counts and a cost estimate."""
        import cv2, base64
        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        tool = {
            "name": "record_inventory",
            "description": "Record every movable household item visible in this room image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "room_type": {"type": "string"},
                    "items": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "size_class": {"type": "string", "enum": list(allowed)},
                            "count":      {"type": "integer", "minimum": 1},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning":  {"type": "string"},
                        },
                        "required": ["size_class", "count", "confidence"],
                    }},
                },
                "required": ["room_type", "items"],
            },
        }
        r = self._c().messages.create(
            model=self.model, max_tokens=2048,
            tools=[tool], tool_choice={"type": "tool", "name": "record_inventory"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT},
            ]}],
        )
        items, room = [], "unknown"
        for blk in r.content:
            if blk.type == "tool_use":
                room = blk.input.get("room_type", "unknown")
                items = [ClassifiedItem(size_class=i["size_class"], count=int(i["count"]),
                                        confidence=float(i.get("confidence", 0.0)),
                                        reasoning=i.get("reasoning", ""))
                         for i in blk.input.get("items", [])]
        pin, pout = PRICING.get(self.model, (2.0, 10.0))
        usage = dict(room_type=room,
                     input_tokens=r.usage.input_tokens,
                     output_tokens=r.usage.output_tokens,
                     cost_usd=round(r.usage.input_tokens / 1e6 * pin
                                    + r.usage.output_tokens / 1e6 * pout, 5))
        return items, usage
