"""
FILE PURPOSE
    Load every result JSON in poc/results/ and rank the combinations, so the sweep
    turns into one table you can make a decision from.

WHY IT RANKS BY BIAS FIRST
    Bias and spread need different fixes and bias is the cheaper problem. A method
    that is consistently 15% low is one multiplier away from correct. A method that is
    randomly +/-15% is not fixable that way. So the table sorts on absolute bias and
    shows spread alongside, rather than collapsing both into one score.

WHAT IT ALSO SHOWS
    - which runs cannot ship (non-commercial models), tagged so they are never
      mistaken for viable options
    - the anchor on/off delta, called out separately because it is the key experiment
    - cost and wall-clock per run, since a marginal accuracy gain at 10x the cost is
      not a gain

USAGE
    python -m poc.runner.compare
    python -m poc.runner.compare --room BED01
    python -m poc.runner.compare --method b --sort abs

HOW TO RUN
    python -m poc.runner.compare
        Rank every result in poc/results/ by bias.
    python -m poc.runner.compare --room BED01
        Only that room.
    python -m poc.runner.compare --method b --sort abs
        Rank Method B instead, sorted by absolute error.
    Run this AFTER run_sweep or several run_combination calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def load(room: str | None):
    rows = []
    for p in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if room and d.get("room") != room:
            continue
        rows.append(d)
    return rows


def fmt(v, w=7, dp=1, suffix=""):
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.{dp}f}{suffix}" if isinstance(v, float) else f"{v:>{w}}{suffix}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rank all combination results.")
    ap.add_argument("--room", default=None)
    ap.add_argument("--method", default="a", choices=["a", "b"], help="which method to rank on")
    ap.add_argument("--sort", default="bias", choices=["bias", "abs", "cost", "time"])
    a = ap.parse_args(argv)

    rows = load(a.room)
    if not rows:
        print(f"No results in {ROOT/'results'}. Run a sweep first:")
        print("  python -m poc.runner.run_sweep --input <file> --room <ROOM>")
        return 1

    key = f"method_{a.method}"
    table = []
    for d in rows:
        s = (d.get("scores") or {}).get(key) or {}
        t = d.get("timings", {})
        table.append(dict(
            tag=d.get("tag") or d["combo_id"][:28],
            detector=d["config"]["detector"],
            depth=d["config"]["depth"],
            mask="yes" if d["config"].get("segmenter") else "-",
            anchor="on" if d["config"].get("scale_anchor") else "OFF",
            ship="" if d.get("shippable") else "  NO-SHIP",
            bias=s.get("vol_bias_pct"),
            absv=s.get("vol_abs_err_pct"),
            prec=s.get("precision"),
            rec=s.get("recall"),
            vol=d["volumes"].get("recognised_m3" if a.method == "a" else "measured_class_m3"),
            cost=d.get("cost_usd", 0.0),
            secs=round(sum(v for v in t.values() if isinstance(v, (int, float))), 1),
        ))

    def sk(r):
        if a.sort == "bias":
            return abs(r["bias"]) if r["bias"] is not None else 1e9
        if a.sort == "abs":
            return r["absv"] if r["absv"] is not None else 1e9
        return -r[{"cost": "cost", "time": "secs"}[a.sort]]
    table.sort(key=sk)

    scored = [r for r in table if r["bias"] is not None]
    print(f"\nMETHOD {a.method.upper()} · {len(table)} runs"
          + (f" · room {a.room}" if a.room else "")
          + f" · {len(scored)} scored against ground truth")
    if not scored:
        print("\n! Nothing is scored — ground_truth.csv is missing or does not cover these rooms.")
        print("  Volumes below are unverified numbers, not accuracy results. Gap A1.\n")

    hdr = (f"{'run':22} {'detector':15} {'depth':11} {'mask':5} {'anch':5} "
           f"{'bias%':>7} {'abs%':>7} {'prec':>5} {'rec':>5} {'m3':>6} {'$':>6} {'s':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in table:
        print(f"{r['tag'][:22]:22} {r['detector'][:15]:15} {r['depth'][:11]:11} "
              f"{r['mask']:5} {r['anchor']:5} "
              f"{fmt(r['bias'])} {fmt(r['absv'])} "
              f"{fmt(r['prec'],5,2)} {fmt(r['rec'],5,2)} {fmt(r['vol'],6)} "
              f"{r['cost']:>6.3f} {r['secs']:>6.1f}{r['ship']}")

    # ---- the key experiment, called out on its own ----
    # Pair runs that differ ONLY by the anchor. Comparing the best anchored run
    # against the best un-anchored run would mix in the detector choice and
    # attribute someone else's gain to the anchor.
    def sig(r):
        return (r["detector"], r["depth"], r["mask"])
    pairs = []
    for r_on in [r for r in table if r["anchor"] == "on" and r["bias"] is not None]:
        for r_off in [r for r in table if r["anchor"] == "OFF" and r["bias"] is not None]:
            if sig(r_on) == sig(r_off):
                pairs.append((r_on, r_off))
    if pairs:
        print(f"\nSCALE ANCHOR — the key experiment")
        print("  matched pairs (identical config, anchor toggled):")
        for r_on, r_off in pairs:
            d = abs(r_off["bias"]) - abs(r_on["bias"])
            print(f"    {sig(r_on)[0]}/{sig(r_on)[1]}: "
                  f"|bias| {abs(r_on['bias']):.1f}% on vs {abs(r_off['bias']):.1f}% off "
                  f"-> anchor worth {d:+.1f} pts")
    else:
        on = [r for r in table if r["anchor"] == "on" and r["bias"] is not None]
        off = [r for r in table if r["anchor"] == "OFF" and r["bias"] is not None]
        if on and off:
            print(f"\nSCALE ANCHOR")
            print(f"  No matched pair found, so this is NOT a clean comparison —")
            print(f"  the two runs differ by more than the anchor.")
            print(f"  best |bias| anchor ON  {min(abs(r['bias']) for r in on):.1f}%")
            print(f"  best |bias| anchor OFF {min(abs(r['bias']) for r in off):.1f}%")
            print(f"  Run the same detector+depth both ways for a real answer.")

    nc = [r for r in table if r["ship"]]
    if nc:
        print(f"\nCANNOT SHIP ({len(nc)}): {', '.join(r['tag'] for r in nc)}")
        print("  Useful as a ceiling measurement. Never as a product decision.")

    print("\nRead bias before abs. Consistent bias is one multiplier from fixed;")
    print("random spread is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
