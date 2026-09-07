"""The deterministic 500-task BFCL v4 subset, and the workload benchmark mode runs.

Artificial Analysis does not publish which 500 of the ~4.4k BFCL v4 tasks it uses, so
we take our own: proportional to each category's share of the suite, and picked by an
even stride over the sorted id list -- no RNG, no seed, reproducible.

`LONG_TAIL` and `dispatch_order` below are what `run --benchmark` adds on top of that:
the same tasks, minus the handful that dominate the wall clock, submitted in an order
that is a pure function of the subset rather than of how fast the endpoint answered.

Imports of `bfcl_eval` are deferred into the functions, because importing that package
creates directories under BFCL_PROJECT_ROOT as a side effect and the caller must be
allowed to set that variable first.
"""

import hashlib
import json
from pathlib import Path

TARGET = 500

# The five tasks that dominate a run's wall clock, slowest first. Measured, not guessed:
# `main.py latency` ranks every task by the per-request latency bfcl-eval records in its
# result rows, summed over every turn and step, and takes the median across runs -- a
# single run ranks the endpoint's bad minutes as much as the suite. These five head that
# ranking at a median of 322-662 s each, against ~6 s for the subset as a whole, and are
# about a fifth of all request-seconds in a run.
#
#   multi_turn_miss_func_147      662 s   16 calls,  72k output tokens
#   multi_turn_long_context_171   499 s   11 calls,  23k
#   simple_java_74                479 s    1 call,   18k
#   multi_turn_base_171           324 s   11 calls,  26k
#   multi_turn_miss_param_171     322 s   14 calls,  45k
#
# Three of them are one scenario, index 171, which the suite repeats once per multi_turn
# category: a three-turn travel-booking-and-messaging session that takes 10-15 model
# calls to finish. `simple_java_74` is the odd one -- a single-turn question about
# serialising an XML surrogate pair, one call, on which the model runs away to tens of
# thousands of output tokens instead of emitting the one function call it is asked for.
# `multi_turn_miss_func_147` is the worst of all and the easiest to miss: it is slow
# enough to hit gateway timeouts, so it failed outright in three of the six runs that
# contain it, and a ranking that reads only completed rows discounts exactly the task
# most worth removing. That is why `latency` prints a fail column.
#
# It is a shoulder, not a cliff -- multi_turn_miss_func_66 is at 283 s and _5 at 237 s.
# Five is where the wall clock stops paying, not where the suite gets cheap.
#
# Dropping them changes the denominator. Every run that does it records the ids in
# config.json and score.json, and a 495-task accuracy must never be compared with a
# 500-task one.
LONG_TAIL = (
    "multi_turn_miss_func_147",
    "multi_turn_long_context_171",
    "simple_java_74",
    "multi_turn_base_171",
    "multi_turn_miss_param_171",
)


def categories():
    # web_search needs a SerpAPI key and memory_* needs a vector / rec-sum backend;
    # neither is available here, so the sample is drawn from the 14 AST- and
    # state-graded categories that make up the bulk of BFCL v4.
    from bfcl_eval.constants.category_mapping import (
        MULTI_TURN_CATEGORY, SINGLE_TURN_CATEGORY)
    return list(SINGLE_TURN_CATEGORY) + list(MULTI_TURN_CATEGORY)


def build(target=TARGET, verbose=True):
    """Return {category: [id, ...]} totalling `target` ids."""
    from bfcl_eval.constants.eval_config import PROMPT_PATH

    cats = categories()
    pool = {}
    for c in cats:
        p = Path(PROMPT_PATH) / f"BFCL_v4_{c}.json"
        pool[c] = sorted(json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip())
    total = sum(len(v) for v in pool.values())

    # Largest-remainder apportionment, with every category guaranteed at least one task.
    exact = {c: len(v) * target / total for c, v in pool.items()}
    alloc = {c: max(1, int(exact[c])) for c in cats}
    while sum(alloc.values()) > target:
        c = max(cats, key=lambda c: (alloc[c] - exact[c], alloc[c]))
        if alloc[c] > 1:
            alloc[c] -= 1
    while sum(alloc.values()) < target:
        c = max(cats, key=lambda c: exact[c] - alloc[c])
        alloc[c] += 1

    subset = {}
    for c in cats:
        ids, n = pool[c], alloc[c]
        subset[c] = [ids[j * len(ids) // n] for j in range(n)]
        assert len(set(subset[c])) == n, c

    if verbose:
        print(f"suite total={total}  subset={sum(len(v) for v in subset.values())}")
        for c in cats:
            print(f"  {c:26s} {len(pool[c]):5d} -> {alloc[c]:3d}")
    return subset


def flatten(subset):
    """{category: [ids]} -> the flat item list stored as subset.json."""
    return [{"id": i, "category": c} for c in sorted(subset) for i in subset[c]]


def group(items):
    """The inverse of flatten, for regenerating the id file from subset.json."""
    out = {}
    for it in items:
        out.setdefault(it["category"], []).append(it["id"])
    return out


def without_long_tail(items, ids=LONG_TAIL):
    """-> (kept, removed, absent).

    `absent` is the ids that were asked for and were not there: a subset built with a
    non-default --target may not contain all five, and the run must say so rather than
    quietly excluding four.
    """
    ids = set(ids)
    kept = [it for it in items if it["id"] not in ids]
    present = {it["id"] for it in items}
    return kept, sorted(ids & present), sorted(ids - present)


def dispatch_order(items):
    """The order bfcl-eval will hand these tasks to its thread pool.

    Not an approximation: `collect_test_cases` ends in `sorted(..., key=sort_key)`, and
    the scheduler seeds *and* refills its pool by popping a heap of that same key, so the
    submission order is fixed by the subset alone -- it does not depend on which task
    finished first, on the endpoint's speed, or on the thread count. `sort_key` reads
    nothing but the id, so a bare {"id": ...} reproduces it exactly.

    The order is (single-turn, then multi-turn) x category x numeric index, which puts
    every expensive multi_turn task in the last fifth of the queue.
    """
    from bfcl_eval.utils import sort_key
    return [it["id"] for it in sorted(items, key=lambda it: sort_key({"id": it["id"]}))]


def order_fingerprint(order):
    """A short digest of a dispatch order, so two runs can be compared at a glance."""
    return hashlib.sha256("\n".join(order).encode()).hexdigest()[:16]
