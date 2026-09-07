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
    Run this AFTER running several pipelines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def result_files() -> list[Path]:
    """Every result JSON, wherever it was written.

    Results live in TWO places since each combination got its own folder:
      poc/results/                    the general-purpose CLI writes here
      poc/pipelines/<name>/results/   a pipeline's own run.py writes here
    This looked only in the first, so after the restructure it reported "no
    results" no matter how many pipelines you had run.
    """
    out = list((ROOT / "results").glob("*.json"))
    out += list((ROOT / "pipelines").glob("*/results/*.json"))
    return sorted(out)


def load(room: str | None):
    rows = []
    for p in result_files():
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if room and d.get("room") != room:
            continue
        # Which pipeline folder produced this, so a 14-way comparison says where
        # each row came from rather than making you match combo_ids by eye.
        d["_pipeline"] = (p.parent.parent.name
                          if p.parent.name == "results" and p.parent.parent.parent.name
                          == "pipelines" else "(general CLI)")
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
        print(f"No results yet. Looked in:")
        print(f"    {ROOT / 'results'}/*.json")
        print(f"    {ROOT / 'pipelines'}/*/results/*.json\n")
        print("Run a pipeline first, for example:")
        print("  python -m poc.pipelines.grounding_dino__moge2__nocls.run --all")
        print("  python -m poc.pipelines.grounding_dino__moge2__sonnet5.run --all")
        print("\nAvailable pipelines:")
        for q in sorted((ROOT / "pipelines").iterdir()):
            if (q / "config.py").is_file():
                print(f"    {q.name}")
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
            # The with/without-LLM arm. Every combination can be run both ways
            # (--classifier none), so this is the column that pairs them up.
            llm=(d["config"].get("classifier") or "none"),
            pipeline=d.get("_pipeline", "?"),
            room=d.get("room", "?"),
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

    # Names differing only in a suffix ("...__sonnet5" vs "...__sonnet5__noanchor")
    # must stay distinguishable, so keep the END of a name that will not fit.
    def short(n, w=34):
        return n if len(n) <= w else "…" + n[-(w - 1):]

    hdr = (f"{'pipeline':34} {'room':14} {'LLM':14} {'mask':5} {'anch':5} "
           f"{'bias%':>7} {'abs%':>7} {'prec':>5} {'rec':>5} {'m3':>7} {'$':>6} {'s':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in table:
        print(f"{short(r['pipeline']):34} {str(r['room'])[:14]:14} {r['llm'][:14]:14} "
              f"{r['mask']:5} {r['anchor']:5} "
              f"{fmt(r['bias'])} {fmt(r['absv'])} "
              f"{fmt(r['prec'],5,2)} {fmt(r['rec'],5,2)} {fmt(r['vol'],7,3)} "
              f"{r['cost']:>6.3f} {r['secs']:>6.1f}{r['ship']}")

    # ---- matched-pair comparisons, one variable at a time ----
    # A pair is only meaningful if EVERYTHING else is equal. The classifier and
    # the room belong in the signature: pairing a with-LLM run against a
    # without-LLM one and calling the difference "the anchor" would credit the
    # anchor with the language model's contribution. Same for the room — two
    # different rooms have different true volumes, so their biases are not
    # comparable at all.
    def sig(r, *ignore):
        f = dict(detector=r["detector"], depth=r["depth"], mask=r["mask"],
                 llm=r["llm"], anchor=r["anchor"], room=r["room"])
        for k in ignore:
            f.pop(k)
        return tuple(sorted(f.items()))

    def matched(field, val_a, val_b, label_a, label_b, title, note):
        """Every pair differing ONLY in `field`."""
        A = [r for r in table if r[field] == val_a and r["bias"] is not None]
        B = [r for r in table if r[field] == val_b and r["bias"] is not None]
        pairs = [(x, y) for x in A for y in B if sig(x, field) == sig(y, field)]
        if not pairs:
            if A and B:
                print(f"\n{title}")
                print(f"  No matched pair — the runs differ by more than {field}, so")
                print(f"  any difference cannot be attributed to it.")
                print(f"  best |bias| {label_a:<12} {min(abs(r['bias']) for r in A):.1f}%")
                print(f"  best |bias| {label_b:<12} {min(abs(r['bias']) for r in B):.1f}%")
                print(f"  {note}")
            return
        print(f"\n{title}")
        print(f"  matched pairs — identical in every other respect:")
        for x, y in pairs:
            d = abs(y["bias"]) - abs(x["bias"])
            print(f"    {x['pipeline'][:34]:34} room {x['room'][:12]:12} "
                  f"|bias| {abs(x['bias']):5.1f}% {label_a} vs {abs(y['bias']):5.1f}% "
                  f"{label_b}  -> {d:+.1f} pts")

    matched("anchor", "on", "OFF", "on", "off",
            "SCALE ANCHOR — the key experiment",
            "Run the same detector+depth+classifier both ways for a real answer.")

    # The axis being tested across all 14 combinations.
    llms = sorted({r["llm"] for r in table})
    for other in [x for x in llms if x != "none"]:
        matched("llm", other, "none", "with", "without",
                f"LANGUAGE MODEL — {other} vs none",
                "Run the same pipeline with and without --classifier for an answer.")
    nc = [r for r in table if r["ship"]]
    if nc:
        print(f"\nCANNOT SHIP ({len(nc)}): "
              f"{', '.join(sorted({r['pipeline'] for r in nc}))}")
        print("  Useful as a ceiling measurement. Never as a product decision.")

    print("\nRead bias before abs. Consistent bias is one multiplier from fixed;")
    print("random spread is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
