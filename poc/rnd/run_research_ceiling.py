"""
FILE PURPOSE
    Run the models we are not allowed to ship, so we can measure what staying
    licence-clean costs us in accuracy. Quarantined output.

WHAT IT RUNS
    yolo_world   AGPL-3.0 via Ultralytics  — is the 20x speed advantage free?
    unidepth2    CC BY-NC 4.0              — best indoor depth in the field

WHY
    One question, answered with a number instead of a guess:
      "How much accuracy are we giving up by only using models we can ship?"
    Small gap  -> the licence costs nothing, question closed.
    Large gap  -> a real input to a real decision (buy the Enterprise Licence,
                  negotiate, or go to hardware depth).

QUARANTINE
    Writes to poc/rnd/results/, never poc/results/. Every record carries
    shippable=false and rnd=true. compare.py cannot see these.

LEGAL
    Ultralytics' published position covers internal R&D, commercial or not.
    Get the client's view before running the YOLO-World combinations. The
    --i-have-legal-clearance flag is deliberately awkward: a speed bump, not
    security.

USAGE
    python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 --dry-run
    python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 \
        --i-have-legal-clearance

HOW TO RUN
    python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 --dry-run
        Show the plan and the licence warnings. Runs nothing.
    python -m poc.rnd.run_research_ceiling --input bedroom.mp4 --room BED01 \
        --i-have-legal-clearance
        Actually run the non-shippable models. Requires the flag by design.
    Output goes to poc/rnd/results/, never poc/results/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC.parent))

from poc.models import registry                                    # noqa: E402
from poc.runner.run_combination import main as run_one              # noqa: E402

RND_RESULTS = POC / "rnd" / "results"
PROD_RESULTS = POC / "results"

# Each entry pairs a research run with the shippable run it should be compared against.
CEILING_RUNS = [
    {
        "id": "ceiling_depth_unidepth2",
        "model": "unidepth2",
        "licence": "CC BY-NC 4.0 — non-commercial, no paid escape",
        "question": "How much indoor depth accuracy does staying MIT/Apache cost us?",
        "compare_against": "moge2 (MIT)",
        "args": ["--detector", "grounding_dino", "--depth", "unidepth2",
                 "--classifier", "claude_sonnet5"],
    },
    {
        "id": "ceiling_det_yolo_world",
        "model": "yolo_world",
        "licence": "AGPL-3.0 — would oblige the client to publish all source, or pay Ultralytics",
        "question": "Is the ~20x speed advantage free, or does detection accuracy drop?",
        "compare_against": "grounding_dino (Apache-2.0)",
        "args": ["--detector", "yolo_world", "--depth", "moge2",
                 "--classifier", "claude_sonnet5"],
    },
    {
        "id": "ceiling_both",
        "model": "yolo_world + unidepth2",
        "licence": "AGPL-3.0 AND CC BY-NC 4.0 — doubly unshippable",
        "question": "What is the absolute ceiling if licences were no object at all?",
        "compare_against": "the best shippable combination",
        "args": ["--detector", "yolo_world", "--depth", "unidepth2",
                 "--classifier", "claude_sonnet5"],
    },
]


def banner():
    print("=" * 78)
    print("  RESEARCH CEILING — MODELS THAT CANNOT SHIP")
    print("=" * 78)
    for r in CEILING_RUNS:
        print(f"\n  [{r['id']}]")
        print(f"    model    {r['model']}")
        print(f"    licence  {r['licence']}")
        print(f"    asks     {r['question']}")
        print(f"    vs       {r['compare_against']}")
    print("\n" + "-" * 78)
    print("  Results go to poc/rnd/results/ and are tagged shippable=false.")
    print("  compare.py never reads them. Do not put these numbers in a plan.")
    print("-" * 78)


def collect(d: Path):
    """Best (lowest |bias|) Method A score from a results directory."""
    best = None
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        s = (r.get("scores") or {}).get("method_a") or {}
        b = s.get("vol_bias_pct")
        if b is None:
            continue
        if best is None or abs(b) < abs(best[1]):
            best = (r.get("tag") or r["combo_id"], b,
                    f"{r['config']['detector']} + {r['config']['depth']}")
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure the research ceiling.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--room", default=None)
    ap.add_argument("--only", default=None, help="comma-separated ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-have-legal-clearance", action="store_true",
                    dest="cleared",
                    help="confirm the client legal has approved running non-shippable models")
    a = ap.parse_args(argv)

    banner()

    runs = CEILING_RUNS
    if a.only:
        want = set(a.only.split(","))
        runs = [r for r in runs if r["id"] in want]

    missing = sorted({m for r in runs for m in
                      (r["args"][1], r["args"][3]) if not registry.is_installed(m)})
    if missing:
        print(f"\n  NOT INSTALLED: {', '.join(missing)}")
        print("  Uncomment the relevant lines in poc/requirements.txt and reinstall.")

    if a.dry_run:
        print("\n(dry run — nothing executed)")
        return 0

    if not a.cleared:
        print("\nSTOPPED. These models cannot ship, and Ultralytics' position covers")
        print("internal R&D. Confirm the client legal has approved this, then re-run with:")
        print("  --i-have-legal-clearance")
        return 2

    RND_RESULTS.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for i, r in enumerate(runs, 1):
        print(f"\n{'='*78}\n[{i}/{len(runs)}] {r['id']}\n{'='*78}")
        argv_run = list(r["args"]) + ["--input", a.input, "--tag", r["id"],
                                      "--allow-noncommercial"]
        if a.room:
            argv_run += ["--room", a.room]
        before = set(PROD_RESULTS.glob("*.json"))
        t0 = time.time()
        try:
            rc = run_one(argv_run)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(r["id"])
            continue
        # run_combination writes into poc/results/ — move it into quarantine and
        # stamp it, so a research number can never surface in a production table.
        new = set(PROD_RESULTS.glob("*.json")) - before
        for p in new:
            d = json.loads(p.read_text())
            d["rnd"] = True
            d["shippable"] = False
            d["rnd_licence"] = r["licence"]
            d["rnd_question"] = r["question"]
            (RND_RESULTS / p.name).write_text(json.dumps(d, indent=2))
            p.unlink()
            print(f"  quarantined -> rnd/results/{p.name}")
        (done if rc == 0 else failed).append(r["id"])
        print(f"  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*78}\nok {len(done)}   failed {len(failed)}")

    # ---- the actual deliverable: what does the licence cost us? ----
    prod, rnd = collect(PROD_RESULTS), collect(RND_RESULTS)
    print("\nLICENCE-CLEAN CEILING")
    if prod and rnd:
        gap = abs(prod[1]) - abs(rnd[1])
        print(f"  best shippable   {prod[2]:32} bias {prod[1]:+6.1f}%")
        print(f"  best research    {rnd[2]:32} bias {rnd[1]:+6.1f}%")
        print(f"  cost of staying licence-clean{'':20} {gap:+6.1f} pts")
        print()
        if gap <= 1.0:
            print("  READ: the licence costs us almost nothing. Question closed —")
            print("        stay clean and stop thinking about it.")
        elif gap <= 4.0:
            print("  READ: a modest gap. Probably not worth a licence negotiation;")
            print("        revisit only if accuracy becomes the binding constraint.")
        else:
            print("  READ: a large gap. Worth a real decision — Enterprise Licence,")
            print("        a negotiation, or hardware depth. Take this to the client.")
    else:
        which = "shippable" if not prod else "research"
        print(f"  Cannot compare — no scored {which} runs found.")
        print("  Run the normal sweep first (needs ground_truth.csv), then this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
