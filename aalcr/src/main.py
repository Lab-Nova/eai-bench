#!/usr/bin/env python3
"""AA-LCR-25 benchmark client (long-context recall).

    run     --endpoint URL --model NAME [--results-dir DIR] [--data-dir DIR]
    pending --results-dir DIR          ids with a response and no grade yet
    show    --results-dir DIR ID       question / gold answer / model response
    record  --results-dir DIR ID 0|1   note on stdin -> grades/<safe_id>.json
    collect --results-dir DIR          -> grades.jsonl + score.json

`run` with no --results-dir mints a new timestamped directory; with one, it resumes
into that directory, retrying every item that has no usable answer.

Grading is not done here. Point the `judge-aalcr` skill at the results directory.
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import endpoint as ep  # noqa: E402
import resultdir  # noqa: E402
from resultdir import ResultDir, is_done  # noqa: E402

COMPONENT = "aalcr"
CLIENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(CLIENT_ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(CLIENT_ROOT, "data")

# Long-context answers are short: the documents are enormous but the reply is a figure
# or a sentence. 24k leaves ample room for reasoning without inviting a runaway.
DEFAULT_MAX_TOKENS = 24576


async def _run_one(client, item, max_tokens):
    row, t0 = ep.timed_row(item)
    result, err = await ep.attempt(
        lambda: client.complete(item["prompt"], max_tokens=max_tokens))
    content, reasoning_len, usage = result if result else ("", 0, None)
    return ep.finish_row(row, t0, response=content, reasoning_len=reasoning_len,
                         usage=usage, error=err)


def cmd_run(a):
    data_dir = a.data_dir or DEFAULT_DATA_DIR

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        subset_rows = rd.read_subset()
        if not subset_rows:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
        items = dataset.rebuild_prompts(data_dir, subset_rows)
    else:
        items = dataset.load_subset(data_dir, n_subset=a.n_subset, limit=a.limit)
        rd = ResultDir.start(RESULTS_BASE)
        rd.write_subset(items)

    # 0 means "no bound": run the whole subset at once. Resolved against the subset
    # rather than the remaining work so that a resume of 3 stragglers records the same
    # concurrency as the run that produced the other 22.
    concurrency = a.concurrency or max(1, len(items))

    incoming = {
        "component": COMPONENT, "mode": "single_turn",
        "endpoint": a.endpoint, "model": a.model,
        "temperature": a.temperature, "top_p": a.top_p,
        "n_subset": len(items), "concurrency": concurrency,
        "max_tokens": a.max_tokens,
        "started_at": rd.config.get("started_at") or resultdir.now_stamp(),
    }
    cfg = rd.reconcile(incoming, force=a.force)

    done_ids, dropped = rd.prepare_resume()
    if dropped:
        print(f"resume: dropped {dropped} incomplete row(s) to failures.jsonl", flush=True)
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"results dir : {rd.path}\n"
          f"prompt chars: min={min(len(i['prompt']) for i in items)} "
          f"max={max(len(i['prompt']) for i in items)}\n"
          f"{len(done_ids)} already done, {len(todo)} to run, concurrency={concurrency}",
          flush=True)
    if not todo:
        print("nothing to do; run `collect` once the grades are in.")
        return

    client = ep.Endpoint(cfg["endpoint"], cfg["model"],
                         temperature=cfg["temperature"], top_p=cfg["top_p"])
    sink = ep.jsonl_writer(rd.responses_path)
    try:
        asyncio.run(ep.drive(todo, lambda it: _run_one(client, it, a.max_tokens),
                             concurrency, sink, label="id"))
    finally:
        sink.close()

    rows = rd.rows_by_id()
    ok = sum(1 for it in items if is_done(rows.get(it["id"]) or {}))
    print(f"\n{ok}/{len(items)} items have an answer. "
          f"Grade with the judge-aalcr skill, then: "
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
    print(f"ID: {a.id}\nCATEGORY: {item.get('document_category')}\n"
          f"DOCUMENT SET: {item.get('document_set_id')}\n")
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
    # Per-category recall is the interesting cut of a long-context benchmark, and the
    # subset deliberately spans all seven categories.
    subset = {str(x["id"]): x for x in rd.read_subset()}
    by_cat = {}
    for g in rd.read_grades().values():
        cat = (subset.get(str(g["id"])) or {}).get("document_category", "?")
        c = by_cat.setdefault(cat, {"correct": 0, "total": 0})
        c["correct"] += int(bool(g.get("correct")))
        c["total"] += 1
    score = rd.collect(COMPONENT, extra={"by_category": by_cat})
    print(json.dumps(score, indent=2))
    if not score["complete"]:
        print(f"\nWARNING: {score['total'] - score['graded']} of {score['total']} items are "
              f"ungraded and counted as incorrect. Run the judge-aalcr skill to finish them.",
              file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume the benchmark")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True)
    r.add_argument("--results-dir", help="resume into this directory instead of creating one")
    r.add_argument("--data-dir", help=f"AA-LCR corpus location (default {DEFAULT_DATA_DIR})")
    r.add_argument("--n-subset", type=int, default=dataset.N_SUBSET)
    r.add_argument("--limit", type=int, help="only run the first N of the subset (smoke test)")
    r.add_argument("--concurrency", type=int, default=0,
                   help="in-flight items; 0 (the default) means the whole subset at once")
    r.add_argument("--temperature", type=float, default=ep.DEFAULT_TEMPERATURE)
    r.add_argument("--top-p", type=float, default=ep.DEFAULT_TOP_P)
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--force", action="store_true",
                   help="resume even though endpoint/model changed (recorded in config.json)")
    r.set_defaults(fn=cmd_run)

    for name, fn, help_ in (("pending", cmd_pending, "ids awaiting a grade"),
                            ("collect", cmd_collect, "grades.jsonl + score.json")):
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
