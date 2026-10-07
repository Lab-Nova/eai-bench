#!/usr/bin/env python3
"""BABILong qa3 client (three-supporting-facts QA hidden in 0k / 128k / 256k / 384k of PG19).

    run      --endpoint URL --model NAME [--results-dir DIR] [--data-dir DIR]
    generate [--data-dir DIR]            fetch / generate the splits, no endpoint
    show     --results-dir DIR ID        question / target / model response
    collect  --results-dir DIR           -> grades/ + grades.jsonl + score.json

`run` with no --results-dir mints a new timestamped directory; with one, it resumes
into that directory, retrying every item that has no usable answer.

Grading is BABILong's own string match, so `run` ends by running `collect` itself.
"""

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import endpoint as ep  # noqa: E402
import resultdir  # noqa: E402
import scoring  # noqa: E402
from resultdir import ResultDir, is_done  # noqa: E402

COMPONENT = "babilong_qa3"
CLIENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(CLIENT_ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(CLIENT_ROOT, "data")

# The answer is one sentence, but qa3 at 256k is reasoned over for ~40k tokens on average
# and the tail reaches the cap: at the suite-wide 131,072 (endpoint.DEFAULT_MAX_TOKENS),
# 3-8 of 100 items per endpoint ran out of budget before answering. Those are counted in
# score.json as `truncated`; "context" asks for the whole window instead.
DEFAULT_MAX_TOKENS = ep.DEFAULT_MAX_TOKENS
DEFAULT_CONTEXT_LEN = 1048576  # GLM-5.3 max_position_embeddings
CONTEXT_MARGIN = 16
_INPUT_TOKENS_RE = re.compile(r"(\d+) tokens from the input messages")
DEFAULT_CONCURRENCY = 50


def max_tokens_arg(s):
    """`--max-tokens` value: an integer cap, or "context" for the whole window."""
    return s if s == "context" else int(s)


def _messages(prompt):
    return [{"role": "system", "content": dataset.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


async def context_budget(client, prompt, context_len):
    """The max_tokens that fills the context window for this prompt (see aalcr/src/main.py)."""
    from openai import BadRequestError
    try:
        stream = await client.create(_messages(prompt), max_tokens=context_len, stream=True)
        async for _ in stream:
            break
        await stream.close()
    except BadRequestError as e:
        m = _INPUT_TOKENS_RE.search(str(e))
        if m:
            return context_len - int(m.group(1)) - CONTEXT_MARGIN
    return context_len - len(prompt) // 3 - CONTEXT_MARGIN


async def complete(client, prompt, max_tokens):
    """Endpoint.complete with BABILong's system message in front of the prompt.

    Kept here rather than in endpoint.py, which is vendored byte-identically across the
    clients and has no system-prompt parameter.
    """
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    content, reasoning, usage, finish = [], [], None, None
    stream = await client.create(_messages(prompt), stream=True,
                                 stream_options={"include_usage": True}, **kwargs)
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        if getattr(choice.delta, "content", None):
            content.append(choice.delta.content)
        reasoning_delta = getattr(choice.delta, "reasoning_content", None)
        if reasoning_delta:
            reasoning.append(reasoning_delta)
    return "".join(content), len("".join(reasoning)), usage, finish


async def _run_one(client, item, max_tokens, context_len):
    row, t0 = ep.timed_row(item)
    budget = max_tokens
    if max_tokens == "context":
        budget, err = await ep.attempt(
            lambda: context_budget(client, item["prompt"], context_len))
        if budget is None:
            return ep.finish_row(row, t0, response="", reasoning_len=0, usage=None,
                                 finish_reason=None, error=err, max_tokens=None)
    result, err = await ep.attempt(lambda: complete(client, item["prompt"], budget))
    content, reasoning_len, usage, finish = result if result else ("", 0, None, None)
    return ep.finish_row(row, t0, response=content, reasoning_len=reasoning_len,
                         usage=usage, interaction_tokens=resultdir.context_lengths([usage])[2],
                         finish_reason=finish, error=err, max_tokens=budget)


def cmd_generate(a):
    data_dir = a.data_dir or DEFAULT_DATA_DIR
    items = dataset.load_subset(data_dir, a.lengths, a.n_per_length)
    for ln in a.lengths:
        chars = [len(it["prompt"]) for it in items if it["length"] == ln]
        print(f"{ln:>6s}  {len(chars)} items, prompt chars {min(chars)}-{max(chars)}")


def cmd_run(a):
    data_dir = a.data_dir or DEFAULT_DATA_DIR

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        subset_rows = rd.read_subset()
        if not subset_rows:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
        items = dataset.rebuild_prompts(data_dir, subset_rows)
    else:
        items = dataset.load_subset(data_dir, a.lengths, a.n_per_length, limit=a.limit)
        rd = ResultDir.start(RESULTS_BASE)
        rd.write_subset(items)

    concurrency = a.concurrency or max(1, len(items))
    incoming = {
        "component": COMPONENT, "mode": "single_turn",
        "endpoint": a.endpoint, "model": a.model,
        "temperature": a.temperature, "top_p": a.top_p,
        "n_subset": len(items), "concurrency": concurrency,
        "system_prompt": dataset.SYSTEM_PROMPT,
        "max_tokens": a.max_tokens, "context_len": a.context_len,
        "started_at": rd.config.get("started_at") or resultdir.now_stamp(),
    }
    cfg = rd.reconcile(incoming, force=a.force)
    # As in aalcr: a cap only truncates, so a resume may change it; log the change.
    for key in ("max_tokens", "context_len"):
        if cfg.get(key) != incoming[key]:
            cfg.setdefault("config_history", []).append(
                {"at": resultdir.now_stamp(),
                 "changed": {key: {"from": cfg.get(key), "to": incoming[key]}}})
            cfg[key] = incoming[key]
            rd.write_config(cfg)

    done_ids, dropped = rd.prepare_resume()
    if dropped:
        print(f"resume: dropped {dropped} incomplete row(s) to failures.jsonl", flush=True)
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"results dir : {rd.path}\n"
          f"{len(done_ids)} already done, {len(todo)} to run, concurrency={concurrency}",
          flush=True)

    if todo:
        client = ep.Endpoint(cfg["endpoint"], cfg["model"],
                             temperature=cfg["temperature"], top_p=cfg["top_p"])
        sink = ep.jsonl_writer(rd.responses_path)
        try:
            asyncio.run(ep.drive(todo,
                                 lambda it: _run_one(client, it, a.max_tokens, a.context_len),
                                 concurrency, sink, label="id"))
        finally:
            sink.close()
    collect(rd)


def cmd_show(a):
    rd = ResultDir.open(a.results_dir)
    item = next((x for x in rd.read_subset() if str(x["id"]) == a.id), None)
    if item is None:
        raise SystemExit(f"id not in this run's subset: {a.id}")
    row = rd.rows_by_id().get(a.id)
    if row is None:
        raise SystemExit(f"no response recorded for {a.id}")
    print(f"ID: {a.id}\n\n=== QUESTION ===\n{item['question']}\n\n=== TARGET ===\n{item['target']}")
    print(f"\n=== MODEL RESPONSE ===\n{row.get('response') or '(empty)'}")
    print(f"\nCORRECT: {int(scoring.compare_answers(item['target'], row.get('response') or '', item['question']))}")


def collect(rd):
    """Grade every answered item, then fold the grades into grades.jsonl and score.json.

    Every answered item is (re)graded on each collect, so the verdicts always reflect the
    current scorer. Unanswered items get no grade and count as incorrect.
    """
    subset = rd.read_subset()
    rows = rd.rows_by_id()
    for item in subset:
        row = rows.get(str(item["id"])) or {}
        if is_done(row):
            ok = scoring.compare_answers(item["target"], row["response"], item["question"])
            rd.record_grade(item["id"], int(ok),
                            scoring.preprocess_output(row["response"]).strip()[:200])

    grades = rd.read_grades()
    by_length = {}
    for item in subset:
        c = by_length.setdefault(item["length"], {"correct": 0, "total": 0})
        c["correct"] += int(bool((grades.get(str(item["id"])) or {}).get("correct")))
        c["total"] += 1
    capped = sum(1 for item in subset
                 if (rows.get(str(item["id"])) or {}).get("finish_reason") == "length")
    score = rd.collect(COMPONENT, extra={"by_length": by_length, "truncated": capped},
                       stratify=lambda r: r.get("length"))

    print(f"\n{'length':>8s} {'correct':>9s} {'acc':>7s}")
    for ln, c in by_length.items():
        print(f"{ln:>8s} {c['correct']:4d}/{c['total']:<4d} {100 * c['correct'] / c['total']:6.1f}%")
    print(f"{'all':>8s} {score['correct']:4d}/{score['total']:<4d} {100 * score['accuracy']:6.1f}%"
          f"   ({score['responded']} answered, {capped} hit max_tokens)")
    print_interaction(score)
    if not score["complete"]:
        print(f"WARNING: {score['total'] - score['graded']} item(s) have no answer and count "
              f"as incorrect; resume with run --results-dir {rd.path}", file=sys.stderr)
    return score


def print_interaction(score):
    """Interaction over answered items per length (a pooled median hides the long ones)."""
    by = score.get("interaction_by") or {}
    if not by:
        return
    print(f"\ninteraction tokens (final context - prompt), answered items only:")
    print(f"{'length':>8s} {'n':>4s} {'median':>14s} {'mean':>8s} {'p90':>8s} {'cap hits':>9s}")
    for ln, b in list(by.items()) + [("all", score)]:
        if b.get("interaction_tokens_median") is None:
            continue
        print(f"{ln:>8s} {b['interaction_tokens_n']:4d} "
              f"{b['interaction_tokens_median']:8,.0f} ± {b['interaction_tokens_median_se']:<5,.0f}"
              f"{b['interaction_tokens_mean']:8,.0f} {b['interaction_tokens_p90']:8,} "
              f"{b['no_answer']['output_limit']:9d}")


def cmd_collect(a):
    print(json.dumps(collect(ResultDir.open(a.results_dir)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def subset_args(s):
        s.add_argument("--data-dir", help=f"splits and generator inputs (default {DEFAULT_DATA_DIR})")
        s.add_argument("--lengths", nargs="+", default=list(dataset.LENGTHS),
                       help="haystack lengths in GPT-2 tokens (default %(default)s)")
        s.add_argument("--n-per-length", type=int, default=dataset.N_PER_LENGTH,
                       help="samples per length, the first N of each split (default %(default)s)")

    r = sub.add_parser("run", help="run or resume the benchmark")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True)
    r.add_argument("--results-dir", help="resume into this directory instead of creating one")
    subset_args(r)
    r.add_argument("--limit", type=int, help="only run the first N items (smoke test)")
    r.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="in-flight items; 0 means all at once (default %(default)s)")
    r.add_argument("--temperature", type=float, default=ep.DEFAULT_TEMPERATURE)
    r.add_argument("--top-p", type=float, default=ep.DEFAULT_TOP_P)
    r.add_argument("--max-tokens", type=max_tokens_arg, default=DEFAULT_MAX_TOKENS,
                   help='per-request cap (default %(default)s, the suite-wide budget), or '
                        '"context": the whole window, context_len - prompt tokens')
    r.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN,
                   help='window size that "context" fills (default %(default)s, GLM-5.3)')
    r.add_argument("--force", action="store_true",
                   help="resume even though endpoint/model changed (recorded in config.json)")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("generate", help="fetch / generate the splits without an endpoint")
    subset_args(g)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("collect", help="grade, then grades.jsonl + score.json")
    s.add_argument("--results-dir", required=True)
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("show", help="print one item and its verdict")
    s.add_argument("--results-dir", required=True)
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
