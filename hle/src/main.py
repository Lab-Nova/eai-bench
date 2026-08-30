#!/usr/bin/env python3
"""HLE-250 benchmark client.

    run     --endpoint URL --model NAME [--tools|--no-tools] [--results-dir DIR]
    pending --results-dir DIR          ids with a response and no grade yet
    show    --results-dir DIR ID       question / gold answer / model response
    record  --results-dir DIR ID 0|1   note on stdin -> grades/<safe_id>.json
    collect --results-dir DIR          -> grades.jsonl + score.json
    audit   --results-dir DIR          contamination scan + integrity check

`run` with no --results-dir mints a new timestamped directory; with one, it resumes
into that directory, retrying every item that has no usable answer.

Grading is not done here. Point the `judge-hle` skill at the results directory.
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import endpoint as ep  # noqa: E402
import runner  # noqa: E402
from resultdir import ResultDir, is_done  # noqa: E402

COMPONENT = "hle"
RESULTS_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")

# Domains that host benchmark question sets (or mirrors of them).
DATASET_DOMAINS = [
    "huggingface.co", "hf.co", "datasets-server.huggingface.co",
    "kaggle.com", "paperswithcode.com", "github.com/centerforaisafety",
    "zenodo.org", "figshare.com",
]
# Narrower: strings that denote the HLE set specifically. A bare "hle" is uselessly
# noisy -- it matches "athlete", the unrelated li-lab/HLE-BioMedX benchmark, and our
# own sandbox temp dirs (/tmp/hle_py_*).
HLE_MARKERS = [
    "hle-public", "hle+public", "cais/hle", "centerforaisafety",
    "humanitys-last-exam", "humanity_s_last_exam", "humanity’s last exam",
]


def cmd_run(a):
    with_tools = a.tools
    mode = "tools" if with_tools else "no_tools"

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        subset_rows = rd.read_subset()
        if not subset_rows:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
        items = dataset.rebuild_prompts(subset_rows, with_tools=with_tools)
        cfg_mode = rd.config.get("mode")
        if cfg_mode and cfg_mode != mode and not a.force:
            raise SystemExit(
                f"{rd.path} was run in mode {cfg_mode!r} but --{'tools' if with_tools else 'no-tools'} "
                f"was given. Pass --force to override, or start a new run.")
    else:
        items = dataset.load_subset(n_subset=a.n_subset, with_tools=with_tools, limit=a.limit)
        rd = ResultDir.start(RESULTS_BASE, suffix=mode.replace("_", ""))
        rd.write_subset(items)

    incoming = {
        "component": COMPONENT, "mode": mode,
        "endpoint": a.endpoint, "model": a.model,
        "temperature": a.temperature, "top_p": a.top_p,
        "n_subset": len(items), "concurrency": a.concurrency,
        "max_tokens": a.max_tokens, "max_rounds": a.max_rounds,
        "started_at": rd.config.get("started_at") or __import__("resultdir").now_stamp(),
    }
    cfg = rd.reconcile(incoming, force=a.force)

    done_ids, dropped = rd.prepare_resume()
    if dropped:
        print(f"resume: dropped {dropped} incomplete row(s) to failures.jsonl", flush=True)
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"results dir : {rd.path}\n"
          f"mode        : {mode}\n"
          f"{len(done_ids)} already done, {len(todo)} to run, concurrency={a.concurrency}",
          flush=True)
    if not todo:
        print("nothing to do; run `collect` once the grades are in.")
        return

    client = ep.Endpoint(cfg["endpoint"], cfg["model"],
                         temperature=cfg["temperature"], top_p=cfg["top_p"])
    if with_tools:
        def work(it):
            return runner.run_with_tools(client, it, max_rounds=a.max_rounds,
                                         gen_budget=a.max_tokens)
    else:
        budget = a.max_tokens or runner.DEFAULT_NOTOOLS_MAX_TOKENS

        def work(it):
            return runner.run_no_tools(client, it, max_tokens=budget)

    sink = ep.jsonl_writer(rd.responses_path)
    try:
        asyncio.run(ep.drive(todo, work, a.concurrency, sink, label="id"))
    finally:
        sink.close()
        if with_tools:
            import tools
            tools.shutdown()

    rows = rd.rows_by_id()
    ok = sum(1 for it in items if is_done(rows.get(it["id"]) or {}))
    print(f"\n{ok}/{len(items)} items have an answer. "
          f"Grade with the judge-hle skill, then: "
          f"python3 {__file__} collect --results-dir {rd.path}")


def cmd_pending(a):
    print(json.dumps(ResultDir.open(a.results_dir).pending_ids(), indent=2))


def cmd_show(a):
    rd = ResultDir.open(a.results_dir)
    item = next((x for x in rd.read_subset() if str(x["id"]) == a.id), None)
    if item is None:
        raise SystemExit(f"id not in this run's subset: {a.id}")
    row = rd.rows_by_id().get(a.id)
    if row is None:
        raise SystemExit(f"no response recorded for {a.id}")
    print(f"ID: {a.id}\nCATEGORY: {item.get('category')}\n"
          f"ANSWER TYPE: {item.get('answer_type')}\n")
    print("=== QUESTION ===")
    print(item["question"])
    print("\n=== GOLD ANSWER ===")
    print(item["gold"])
    print("\n=== MODEL RESPONSE ===")
    print(row.get("response") or "(empty)")


def cmd_record(a):
    note = a.note if a.note is not None else sys.stdin.read().strip()
    g = ResultDir.open(a.results_dir).record_grade(a.id, a.correct, note)
    print(json.dumps(g, ensure_ascii=False))


def cmd_collect(a):
    rd = ResultDir.open(a.results_dir)
    score = rd.collect(COMPONENT)
    print(json.dumps(score, indent=2))
    if not score["complete"]:
        print(f"\nWARNING: {score['total'] - score['graded']} of {score['total']} items are "
              f"ungraded and counted as incorrect. Run the judge-hle skill to finish them.",
              file=sys.stderr)


def cmd_audit(a):
    """Contamination scan plus a consistency check of the stored artifacts."""
    rd = ResultDir.open(a.results_dir)
    rows = rd.read_rows()
    subset = {str(x["id"]) for x in rd.read_subset()}
    grades = rd.read_grades()
    ids = [str(r.get("id")) for r in rows]

    print("=== integrity ===")
    print(f"rows {len(rows)}  unique {len(set(ids))}  == subset: {set(ids) == subset}")
    print(f"missing from responses: {sorted(subset - set(ids))[:10] or 'none'}")
    print(f"graded {len(grades)}  correct {sum(1 for g in grades.values() if g.get('correct'))}")
    print(f"ungraded ids: {[i for i in ids if i not in grades][:10] or 'none'}")
    bad = [r["id"] for r in rows
           if grades.get(str(r["id"]), {}).get("correct") and not (r.get("response") or "").strip()]
    print(f"marked-correct-but-empty: {len(bad)} {bad}")
    print(f"rows with error set: {sum(1 for r in rows if r.get('error'))}")
    print(f"forced_final: {sum(1 for r in rows if r.get('forced_final'))}  "
          f"context_full: {sum(1 for r in rows if r.get('context_full'))}")

    # With web search enabled the model can in principle retrieve the question set
    # itself. Official "HLE w/ tools" numbers carry the same exposure, but it should be
    # quantified before publishing rather than assumed absent.
    print("\n=== contamination ===")
    dom_hits, hle_hits, searched = {}, [], 0
    for r in rows:
        parts = []
        for t in (r.get("tool_trace") or []):
            parts.append(json.dumps(t, ensure_ascii=False))
            if t.get("tool") == "web_search":
                searched += 1
        blob = "\n".join(parts).lower()
        if not blob:
            continue
        hit_domain = False
        for d in DATASET_DOMAINS:
            if d in blob:
                dom_hits.setdefault(d, []).append(r["id"])
                hit_domain = True
        if hit_domain and any(m in blob for m in HLE_MARKERS):
            hle_hits.append(r["id"])
    print(f"web_search tool calls: {searched}")
    for d, hit in sorted(dom_hits.items(), key=lambda kv: -len(kv[1])):
        print(f"  {d:35s} {len(hit):4d} rows")
    if not dom_hits:
        print("  no dataset-hosting domain appeared in any tool trace")
    print(f"rows touching a dataset host AND an HLE marker: {len(hle_hits)}")
    for i in hle_hits[:20]:
        print("  ", i)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume the benchmark")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://host:30000/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--results-dir", help="resume into this directory instead of creating one")
    mode = r.add_mutually_exclusive_group()
    mode.add_argument("--tools", dest="tools", action="store_true", default=True,
                      help="agentic mode with python + web_search (default)")
    mode.add_argument("--no-tools", dest="tools", action="store_false",
                      help="single-turn, no tools")
    r.add_argument("--n-subset", type=int, default=dataset.N_SUBSET)
    r.add_argument("--limit", type=int, help="only run the first N of the subset (smoke test)")
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--temperature", type=float, default=ep.DEFAULT_TEMPERATURE)
    r.add_argument("--top-p", type=float, default=ep.DEFAULT_TOP_P)
    r.add_argument("--max-tokens", type=int, default=0,
                   help="generation cap; 0 = uncapped in tools mode, 65536 in no-tools mode")
    r.add_argument("--max-rounds", type=int, default=0, help="tool-loop cap; 0 = uncapped")
    r.add_argument("--force", action="store_true",
                   help="resume even though endpoint/model/mode changed (recorded in config.json)")
    r.set_defaults(fn=cmd_run)

    for name, fn, help_ in (("pending", cmd_pending, "ids awaiting a grade"),
                            ("collect", cmd_collect, "grades.jsonl + score.json"),
                            ("audit", cmd_audit, "contamination + integrity")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--results-dir", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("show", help="print one item for grading")
    s.add_argument("--results-dir", required=True)
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("record", help="record one verdict")
    s.add_argument("--results-dir", required=True)
    s.add_argument("id")
    s.add_argument("correct", type=int, choices=(0, 1))
    s.add_argument("--note", help="one-sentence justification; read from stdin if omitted")
    s.set_defaults(fn=cmd_record)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
