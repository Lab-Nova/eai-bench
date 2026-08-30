#!/usr/bin/env python3
"""Endpoint Accuracy Index: combine the three components into one composite.

    python3 synthesis.py                          # latest scored run per client
    python3 synthesis.py --hle DIR --aalcr DIR --bfcl DIR
    python3 synthesis.py --json out.json

Reads only each run's `score.json`, never raw responses, so the three clients stay
independent of this script and of each other.

The index is the equally-weighted mean of three accuracies:

    BFCL v4 (500 tasks, machine-graded)
    HLE     (250 questions, LLM-judged)
    AA-LCR  ( 25 questions, LLM-judged)

The BFCL subscore is a *flat* pass rate over the sampled tasks, not bfcl-eval's own
category-weighted "Overall Acc" -- a flat rate is what "accuracy on 500 tasks" means
and keeps all three components on the same footing. The weighted figure is printed
underneath as a footnote.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENTS = ("bfcl", "hle", "aalcr")
LABELS = {"bfcl": "BFCL-500", "hle": "HLE-250", "aalcr": "AA-LCR-25"}


def load_score(path):
    p = os.path.join(path, "score.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def latest_scored(client, mode=None):
    """The lexically greatest run under <client>/results that has a score.json.

    Timestamps are UTC and zero-padded, so lexical order is chronological order.
    """
    base = os.path.join(HERE, client, "results")
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base), reverse=True):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        s = load_score(d)
        if s is None:
            continue
        if mode is not None and s.get("mode") != mode:
            continue
        return d
    return None


def pct(score):
    return 100.0 * score["accuracy"]


def fmt_component(label, s, marker=""):
    if s is None:
        return f"  {label:16s} {'not run':>16s}"
    flag = "" if s.get("complete", True) else "  INCOMPLETE"
    return (f"  {label + marker:16s} {s['correct']:4d}/{s['total']:<4d} "
            f"{pct(s):6.2f}%{flag}")


def runtime_line(s):
    bits = []
    if s.get("latency_s_median") is not None:
        bits.append(f"latency med {s['latency_s_median']:.0f}s / "
                    f"p90 {s['latency_s_p90']:.0f}s")
    if s.get("completion_tokens_total"):
        bits.append(f"{s['completion_tokens_total']:,} completion tokens")
    if s.get("prompt_tokens_total"):
        bits.append(f"{s['prompt_tokens_total']:,} prompt tokens")
    if s.get("rounds_median") is not None:
        bits.append(f"rounds med {s['rounds_median']:g} / max {s['rounds_max']}")
    if s.get("tool_calls_total"):
        bits.append(f"{s['tool_calls_total']:,} tool calls")
    return "  " + ", ".join(bits) if bits else ""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in CLIENTS:
        p.add_argument(f"--{c}", help=f"{c} results directory (default: latest scored)")
    p.add_argument("--hle-no-tools", help="an HLE no-tools run to report as a reference")
    p.add_argument("--json", help="also write the summary here")
    p.add_argument("--allow-partial", action="store_true",
                   help="print a composite even with a component missing")
    a = p.parse_args()

    dirs = {c: getattr(a, c) or latest_scored(c, mode="tools" if c == "hle" else None)
            for c in CLIENTS}
    # An HLE run that used no tools is still a valid HLE run; fall back to any scored
    # run if the with-tools preference found nothing.
    if dirs["hle"] is None:
        dirs["hle"] = latest_scored("hle")
    scores = {c: (load_score(d) if d else None) for c, d in dirs.items()}
    ref_dir = a.hle_no_tools or latest_scored("hle", mode="no_tools")
    ref = load_score(ref_dir) if ref_dir else None
    if ref is not None and dirs["hle"] == ref_dir:
        ref = None  # the headline run is the no-tools one; do not print it twice

    print("=" * 66)
    print("Endpoint Accuracy Index -- components")
    print("=" * 66)
    endpoints = set()
    for c in CLIENTS:
        s = scores[c]
        marker = " *" if c == "hle" and s and s.get("mode") == "tools" else ""
        print(fmt_component(LABELS[c], s, marker))
        if s:
            rl = runtime_line(s)
            if rl:
                print(f"  {'':16s}{rl}")
            if s.get("endpoint"):
                endpoints.add((s.get("endpoint"), s.get("model")))
            print(f"  {'':16s}  {dirs[c]}")
    if any(s and s.get("mode") == "tools" for s in scores.values()):
        print("  * HLE with tools -- the setting the official methodology reports")
    if ref is not None:
        print(fmt_component("HLE-250 (ref)", ref) + "   no tools, reference only")

    if len(endpoints) > 1:
        print("\nWARNING: the components were not all measured against the same endpoint:")
        for url, model in sorted(endpoints):
            print(f"  {model} @ {url}")

    have = [c for c in CLIENTS if scores[c] is not None]
    missing = [c for c in CLIENTS if scores[c] is None]
    incomplete = [c for c in have if not scores[c].get("complete", True)]

    print("\n" + "=" * 66)
    if missing and not a.allow_partial:
        # score_index.py averaged over only the components it found, so a two-of-three
        # run printed a mean of two under a heading that said three. Refuse instead.
        print(f"NO COMPOSITE: {', '.join(missing)} has no scored run.")
        print("Run the missing component, or pass --allow-partial to average what "
              "is here (the result is labelled PARTIAL and is not the index).")
        composite = None
    else:
        composite = 100.0 * sum(scores[c]["accuracy"] for c in have) / len(have)
        tag = "" if not missing else f"   PARTIAL ({len(have)}/3)"
        print(f"Equally-weighted composite: {composite:.2f} / 100{tag}")
        if incomplete:
            print(f"WARNING: {', '.join(incomplete)} has ungraded items, counted as "
                  f"incorrect -- this composite is a floor, not a final number.")
    print("=" * 66)

    native = (scores["bfcl"] or {}).get("native_weighted_overall")
    if native and native.get("overall_acc"):
        print(f"\nFor reference, bfcl-eval's own category-weighted Overall Acc: "
              f"{native['overall_acc']}")
        print(f"  ({native['note']})")

    if a.json:
        out = {"composite": composite,
               "components": {c: scores[c] for c in CLIENTS},
               "results_dirs": dirs,
               "hle_no_tools_reference": ref}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")

    return 0 if composite is not None else 1


if __name__ == "__main__":
    sys.exit(main())
