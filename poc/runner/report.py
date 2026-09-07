"""
FILE PURPOSE
    Human-readable output for a pipeline run: annotated images, a printed item
    table, and a CSV. The results JSON is for machines and for compare.py; this
    module is for the person deciding whether to trust the numbers.

WHAT IT PRODUCES
    1. frame_NNN.jpg  - the frame with a box per object, labelled with the class,
       the measured W x D x H and the volume. The YOLO-style picture.
    2. items.csv      - one row per measured object, plus the inventory totals.
       CSV because a removals surveyor reviews this in Excel, not in a JSON viewer.
    3. a console table - counts, unit volume, total volume, grand total.

WHY IT IS A SEPARATE MODULE
    Both run_combination (CLI) and the step-by-step notebook draw the same
    picture. Duplicating the drawing code in the notebook is how a notebook
    starts quietly disagreeing with the pipeline it is supposed to illustrate,
    so both import annotate() from here.

COLOUR IS INFORMATION, NOT DECORATION
    green  measured outright - depth was genuinely visible
    amber  depth was NOT visible; the cube table supplied it (a class prior)
    red    detected but unnamed - the size class came from geometry alone
    blue   the door: a measuring reference, deliberately not counted as cargo

    So a picture full of amber and red means the volume rests on assumptions,
    and you can see that at a glance instead of digging through the JSON.

USED BY  poc/runner/run_combination.py, poc/grounding_dino__moge2__stepbystep.ipynb
"""
from __future__ import annotations

import csv
from pathlib import Path

GREEN, AMBER, RED, BLUE, GREY = ((60, 170, 60), (30, 160, 240), (60, 60, 220),
                                 (220, 150, 40), (150, 150, 150))


def _colour(det_label: str, depth_source: str) -> tuple[int, int, int]:
    if not det_label:
        return RED
    return GREEN if depth_source == "observed" else AMBER


def _band(img, x, y, text, colour, cv2, scale=0.5, pad=4):
    """Text on a filled bar, so a label stays readable over any wallpaper."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + pad * 2, y)
    x = min(x, img.shape[1] - tw - pad * 2 - 1)
    x = max(0, x)
    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y), colour, -1)
    cv2.putText(img, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return th + pad * 2


def annotate(img, pairs, *, doors=(), scale=None, total_m3=None, title=""):
    """Draw one frame's measured objects.

    pairs  [(Detection, measurement row)] - the row carries w/d/h, volume,
           mapped_class and depth_source, so the label can state what was
           measured rather than only what was recognised.
    doors  Detections used as the scale anchor, drawn but not counted.
    """
    import cv2
    vis = img.copy()

    for d in doors:
        x0, y0, x1, y1 = (int(v) for v in d.box)
        cv2.rectangle(vis, (x0, y0), (x1, y1), BLUE, 2)
        _band(vis, x0, y0, "door - scale reference, not cargo", BLUE, cv2, 0.45)

    for d, r in pairs:
        x0, y0, x1, y1 = (int(v) for v in d.box)
        col = _colour(r.get("det_label", ""), r.get("depth_source", ""))
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        name = r.get("mapped_class") or r.get("det_label") or "unnamed"
        h = _band(vis, x0, y0, f"{name}   {r['bbox_m3']:.2f} m3", col, cv2, 0.5)
        _band(vis, x0, y0 + h + 1,
              f"{r['w']:.2f} x {r['d']:.2f} x {r['h']:.2f} m"
              + ("" if r.get("depth_source") == "observed" else "  (depth assumed)"),
              col, cv2, 0.42)

    head = [t for t in (title,
                        f"{len(pairs)} objects",
                        None if total_m3 is None else f"total {total_m3:.2f} m3",
                        None if scale is None else f"scale x{scale:.4f}") if t]
    if head:
        _band(vis, 0, vis.shape[0] - 1, "   ".join(head), (40, 40, 40), cv2, 0.5)
    return vis


def write_visuals(frames, out_dir: Path, *, scale=None, total_m3=None) -> list[Path]:
    """One annotated JPEG per frame. Returns what it wrote."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, f in enumerate(frames):
        pairs = f.get("pairs", [])
        doors = [d for d in f.get("dets", []) if d.label == "door"]
        vis = annotate(f["img"], pairs, doors=doors, scale=scale,
                       total_m3=total_m3, title=f"frame {i}")
        p = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        written.append(p)
    return written


def item_lines(rows, inv, classes) -> list[str]:
    """The console report: per-object measurements, then the inventory."""
    out = ["", "  MEASURED OBJECTS" + (" (none)" if not rows else ""),
           f"  {'object':<20} {'W':>6} {'D':>6} {'H':>6} {'m3':>7}  {'size class':<20} depth"]
    out.append("  " + "-" * 78)
    for r in sorted(rows, key=lambda r: -r["bbox_m3"]):
        out.append(f"  {(r.get('det_label') or '(unnamed)'):<20} "
                   f"{r['w']:>6.2f} {r['d']:>6.2f} {r['h']:>6.2f} {r['bbox_m3']:>7.3f}  "
                   f"{(r.get('mapped_class') or '-'):<20} {r.get('depth_source','')}")

    out += ["", "  INVENTORY (what would be quoted)",
            f"  {'size class':<24} {'count':>5} {'each m3':>9} {'total m3':>9}"]
    out.append("  " + "-" * 52)
    grand = 0.0
    for cls in sorted(inv):
        each = classes.get(cls, {}).get("cube_m3")
        if each is None:
            out.append(f"  {cls:<24} {inv[cls]:>5} {'?':>9} {'?':>9}   not in cube_table")
            continue
        tot = each * inv[cls]
        grand += tot
        out.append(f"  {cls:<24} {inv[cls]:>5} {each:>9.3f} {tot:>9.3f}")
    out.append("  " + "-" * 52)
    out.append(f"  {'TOTAL':<24} {sum(inv.values()):>5} {'':>9} {grand:>9.3f}")
    return out


def write_csv(rows, inv, classes, path: Path) -> Path:
    """Per-object rows then the inventory, in one file a surveyor can open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "object_or_class", "count", "w_m", "d_m", "h_m",
                    "volume_m3", "size_class", "depth_source", "score"])
        for r in sorted(rows, key=lambda r: -r["bbox_m3"]):
            w.writerow(["measurement", r.get("det_label") or "(unnamed)", 1,
                        f"{r['w']:.3f}", f"{r['d']:.3f}", f"{r['h']:.3f}",
                        f"{r['bbox_m3']:.3f}", r.get("mapped_class", ""),
                        r.get("depth_source", ""), r.get("score", "")])
        grand = 0.0
        for cls in sorted(inv):
            each = classes.get(cls, {}).get("cube_m3")
            tot = (each or 0) * inv[cls]
            grand += tot
            w.writerow(["inventory", cls, inv[cls], "", "", "",
                        f"{tot:.3f}", cls, "", ""])
        w.writerow(["total", "ALL", sum(inv.values()), "", "", "", f"{grand:.3f}", "", "", ""])
    return path
