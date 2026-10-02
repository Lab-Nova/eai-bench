#!/usr/bin/env python3
"""Counting-Stars client (long-context multi-needle correction test, EN + ZH, 64k-960k).

The standard run is 2 languages x 15 lengths x 5 samples = 150 requests.

    run      --endpoint URL --model NAME [--results-dir DIR] [--data-dir DIR]
    generate [--data-dir DIR]            build and cache the prompts, no endpoint
    show     --results-dir DIR ID        reference / wrong counts / model response
    collect  --results-dir DIR           -> grades.jsonl + score.json

`run` with no --results-dir mints a new timestamped directory; with one, it resumes
into that directory, retrying every item that has no usable answer.

Grading is mechanical, so `run` ends by running `collect` itself.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import endpoint as ep  # noqa: E402
import resultdir  # noqa: E402
import scoring  # noqa: E402
from resultdir import ResultDir, is_done  # noqa: E402

COMPONENT = "counting_stars"
CLIENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(CLIENT_ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(CLIENT_ROOT, "data")

# The answer is a 32-number list, but at 960k the model reasons for tens of thousands of
# tokens first. The default is the suite-wide 131,072 (endpoint.DEFAULT_MAX_TOKENS);
# "context" asks for the whole window, max_tokens = context_len - prompt tokens.
DEFAULT_MAX_TOKENS = ep.DEFAULT_MAX_TOKENS
DEFAULT_CONTEXT_LEN = 1048576  # GLM-5.3 max_position_embeddings
CONTEXT_MARGIN = 16
# An integer cap is lowered to fit prompt + output in the window: at 960k the prompt is
# ~983k tokens, and sglang refuses prompt + max_tokens over context_len outright rather
# than stopping at the window. 512 covers the chat template on top of prompt_tokens, as
# in the original Counting-Stars runner.
TEMPLATE_MARGIN = 512
_INPUT_TOKENS_RE = re.compile(r"(\d+) tokens from the input messages")
# 150 requests of up to a million tokens each: the server's KV pool, not this flag, is
# the real limit, and far past it extra requests only queue prefills behind each other.
DEFAULT_CONCURRENCY = 16


def max_tokens_arg(s):
    """`--max-tokens` value: an integer cap, or "context" for the whole window."""
    return s if s == "context" else int(s)


async def context_budget(client, prompt, context_len):
    """The max_tokens that fills the context window for this prompt (see aalcr/src/main.py).

    An sglang server refuses max_tokens = the whole window with "... X tokens from the
    input messages ...", which is the exact prompt size after its chat template. Without
    a count, fall back to a chars/2 estimate.
    """
    from openai import BadRequestError
    messages = [{"role": "user", "content": prompt}]
    try:
        stream = await client.create(messages, max_tokens=context_len, stream=True)
        async for _ in stream:
            break
        await stream.close()
    except BadRequestError as e:
        m = _INPUT_TOKENS_RE.search(str(e))
        if m:
            return context_len - int(m.group(1)) - CONTEXT_MARGIN
    return context_len - len(prompt) // 2 - CONTEXT_MARGIN


async def _run_one(client, item, max_tokens, context_len):
    row, t0 = ep.timed_row(item, drop=("prompt", "reference", "wrong"))
    budget = max_tokens
    if isinstance(budget, int) and budget and item.get("prompt_tokens"):
        budget = min(budget, context_len - item["prompt_tokens"] - TEMPLATE_MARGIN)
    if max_tokens == "context":
        budget, err = await ep.attempt(
            lambda: context_budget(client, item["prompt"], context_len))
        if budget is None:
            return ep.finish_row(row, t0, response="", reasoning_len=0, usage=None,
                                 error=err, max_tokens=None)
    result, err = await ep.attempt(
        lambda: client.complete(item["prompt"], max_tokens=budget))
    content, reasoning_len, usage = result if result else ("", 0, None)
    return ep.finish_row(row, t0, response=content, reasoning_len=reasoning_len,
                         usage=usage, interaction_tokens=resultdir.context_lengths([usage])[2],
                         error=err, max_tokens=budget)


def cmd_generate(a):
    data_dir = a.data_dir or DEFAULT_DATA_DIR
    for it in dataset.load_subset(data_dir, a.langs, a.lengths, n_samples=1):
        print(f"{it['lang']}-{it['length']:>5s}  {it['prompt_tokens']:8d} tokens")


def cmd_run(a):
    data_dir = a.data_dir or DEFAULT_DATA_DIR

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        subset_rows = rd.read_subset()
        if not subset_rows:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
        items = dataset.rebuild_prompts(data_dir, subset_rows)
    else:
        items = dataset.load_subset(data_dir, a.langs, a.lengths, a.n_samples, limit=a.limit)
        rd = ResultDir.start(RESULTS_BASE)
        rd.write_subset(items)

    concurrency = a.concurrency or max(1, len(items))
    incoming = {
        "component": COMPONENT, "mode": dataset.TASK,
        "endpoint": a.endpoint, "model": a.model,
        "temperature": a.temperature, "top_p": a.top_p,
        "n_subset": len(items), "concurrency": concurrency,
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
    print(f"ID: {a.id}   prompt tokens: {item['prompt_tokens']}\n")
    print(f"CORRECT COUNTS: {item['reference']}")
    print(f"WRONG COUNTS  : {item['wrong']}\n")
    print("=== MODEL RESPONSE ===")
    print(row.get("response") or "(empty)")
    print("\n=== SCORE ===")
    print(json.dumps(scoring.grade(row.get("response"), item)))


def _mean(xs):
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def collect(rd):
    """Score every item in subset order into grades.jsonl and score.json.

    The denominator is the subset: an item with no usable answer scores 0, it is not
    excused. `accuracy` is the mean per-prompt score, so `correct` is a fractional sum.
    """
    subset = rd.read_subset()
    rows = rd.rows_by_id()
    grades = []
    for item in subset:
        row = rows.get(str(item["id"])) or {}
        answered = is_done(row)
        g = {"id": item["id"], "lang": item["lang"], "length": item["length"],
             "sample": item["sample"], "responded": answered,
             **scoring.grade(row.get("response") if answered else "", item)}
        grades.append(g)
    with open(rd.grades_path, "w", encoding="utf-8") as f:
        for g in grades:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    total = len(subset)
    responded = sum(g["responded"] for g in grades)
    by_lang = {lang: _mean([g["score"] for g in grades if g["lang"] == lang])
               for lang in dict.fromkeys(g["lang"] for g in grades)}
    by_length = {ln: _mean([g["score"] for g in grades if g["length"] == ln])
                 for ln in dict.fromkeys(g["length"] for g in grades)}
    # Samples of one prompt are not independent of each other, so the standard error is
    # taken over the (lang, length) cell means, not over the individual requests.
    cells = {}
    for g in grades:
        cells.setdefault((g["lang"], g["length"]), []).append(g["score"])
    cell_means = [sum(v) / len(v) for v in cells.values()]
    se = (statistics.stdev(cell_means) / len(cell_means) ** 0.5) if len(cell_means) > 1 else 0.0
    score = {
        "component": COMPONENT,
        "correct": round(sum(g["score"] for g in grades), 6),
        "total": total,
        "accuracy": _mean([g["score"] for g in grades]),
        "accuracy_se": round(se, 6),
        "n_cells": len(cells),
        "graded": total,
        "responded": responded,
        # Mechanical grading leaves nothing to judge; a run is complete once every
        # item has an answer.
        "complete": responded == total,
        "results_dir": rd.path,
        "scored_at": resultdir.now_stamp(),
        "by_lang": by_lang,
        "by_length": by_length,
        "needles": {k: sum(g[k] for g in grades)
                    for k in ("n_correct", "n_both", "n_wrong", "n_missing")},
    }
    score.update(rd.runtime_stats())
    cfg = rd.config
    for k in ("endpoint", "model", "mode"):
        if k in cfg:
            score[k] = cfg[k]
    rd._write_json(rd.score_path, score)

    print(f"\nscore = mean over {scoring.M} needles "
          f"(1 correct only, 0.5 both, 0.25 wrong only), averaged over samples")
    langs = list(by_lang)
    print(f"{'length':>8s} " + " ".join(f"{x:>6s}" for x in langs))
    for ln in by_length:
        print(f"{ln:>8s} " + " ".join(
            f"{_mean(cells[(x, ln)]):6.3f}" if (x, ln) in cells else f"{'-':>6s}" for x in langs))
    print(f"{'mean':>8s} " + " ".join(f"{by_lang[x]:6.3f}" for x in langs)
          + f"   overall {score['accuracy']:.3f} +- {se:.3f}  ({responded}/{total} answered)")
    if score.get("interaction_tokens_median") is not None:
        print(f"interaction tokens (final context - prompt): "
              f"median {score['interaction_tokens_median']:,.0f}  "
              f"p90 {score['interaction_tokens_p90']:,}  max {score['interaction_tokens_max']:,}")
    if not score["complete"]:
        print(f"WARNING: {total - responded} item(s) have no answer and score 0; "
              f"resume with run --results-dir {rd.path}", file=sys.stderr)
    return score


def cmd_collect(a):
    collect(ResultDir.open(a.results_dir))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def subset_args(s):
        s.add_argument("--data-dir", help=f"inputs and prompt cache (default {DEFAULT_DATA_DIR})")
        s.add_argument("--langs", nargs="+", choices=dataset.LANGS, default=list(dataset.LANGS))
        s.add_argument("--lengths", nargs="+", default=list(dataset.LENGTHS),
                       help="prompt lengths in GLM tokens (default 64k 128k ... 960k)")
        s.add_argument("--n-samples", type=int, default=dataset.N_SAMPLES,
                       help="independent samples of each (lang, length) prompt "
                            "(default %(default)s)")

    r = sub.add_parser("run", help="run or resume the benchmark")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True)
    r.add_argument("--results-dir", help="resume into this directory instead of creating one")
    subset_args(r)
    r.add_argument("--limit", type=int, help="only run the first N prompts (smoke test)")
    r.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="in-flight prompts; 0 means all at once (default %(default)s)")
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

    g = sub.add_parser("generate", help="build and cache the prompts without an endpoint")
    subset_args(g)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("collect", help="grades.jsonl + score.json")
    s.add_argument("--results-dir", required=True)
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("show", help="print one item and its score")
    s.add_argument("--results-dir", required=True)
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
