"""The deterministic 500-task BFCL v4 subset.

Artificial Analysis does not publish which 500 of the ~4.4k BFCL v4 tasks it uses, so
we take our own: proportional to each category's share of the suite, and picked by an
even stride over the sorted id list -- no RNG, no seed, reproducible.

Imports of `bfcl_eval` are deferred into the functions, because importing that package
creates directories under BFCL_PROJECT_ROOT as a side effect and the caller must be
allowed to set that variable first.
"""

import json
from pathlib import Path

TARGET = 500


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
