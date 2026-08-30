# eai-bench — Endpoint Accuracy Index

Three benchmark clients that measure one OpenAI-compatible endpoint, plus a synthesis
step that combines them into a single number.

| Component | Size | Grading | What it measures |
|---|---|---|---|
| `bfcl/` | 500 tasks | machine (`bfcl-eval`, AST + state) | function / tool calling |
| `hle/` | 250 questions | LLM judge | frontier reasoning (Humanity's Last Exam) |
| `aalcr/` | 25 questions | LLM judge | long-context recall (AA-LCR) |

The index is the **equally-weighted mean of the three accuracies**. Each client is
self-contained: it takes `--endpoint` and `--model`, writes one timestamped results
directory, and resumes into that directory if interrupted.

```
eai-bench/
  README.md
  synthesis.py            latest-or-specified results per client -> composite
  hle/  aalcr/  bfcl/     one benchmark client each: src/, results/, README.md
  .claude/skills/         judge-hle, judge-aalcr -- the LLM grading procedures
  tools/                  import_legacy.py, selftest.py
  serving/                the sglang launch script the reference numbers used
```

## Running all three

Point every client at the same endpoint. Nothing here starts or stops a server; on a
shared GPU box, launch one through `gpu-run` first (see `serving/`).

```bash
EP=http://127.0.0.1:30000/v1
MODEL=GLM-5.3

python3 bfcl/src/main.py  setup                              # once: builds bfcl/.venv
python3 bfcl/src/main.py  run --endpoint $EP --model $MODEL
python3 bfcl/src/main.py  collect --results-dir bfcl/results/<ts>

python3 hle/src/main.py   run --endpoint $EP --model $MODEL --tools
#   ... then grade it with the judge-hle skill, which ends by running `collect`

python3 aalcr/src/main.py run --endpoint $EP --model $MODEL
#   ... then grade it with the judge-aalcr skill

python3 synthesis.py
```

`synthesis.py` with no arguments picks the latest scored run under each client (for HLE
it prefers a with-tools run and reports a no-tools run as a labelled reference). Pass
`--hle DIR --aalcr DIR --bfcl DIR` to pin exact runs, `--json OUT` to save the summary.

It **refuses to print a composite unless all three components are present**. Its
predecessor averaged over whatever it found, so a two-of-three run silently produced a
mean of two under a heading that said three. Use `--allow-partial` to get that number
anyway; it is labelled `PARTIAL` and is not the index.

## A results directory

Every run is one directory, and everything about it lives inside:

```
<client>/results/2026-08-29T22-22-14Z-tools/
  config.json      endpoint, model, sampling, mode, subset size, started_at
  subset.json      the exact items chosen -- the run's denominator
  responses.jsonl  one row per item, appended and flushed as it completes
  failures.jsonl   rows a resume dropped, kept as an audit trail
  grades/<id>.json one verdict per item, written by the judge skill
  grades.jsonl     collected, in subset order
  score.json       the uniform summary synthesis.py reads
```

Timestamps are UTC and filesystem-safe, so they sort lexically and "latest" is just
`max()`. `results/` is git-ignored: it is large, per-run, and reproducible.

`score.json` is the only contract between a client and `synthesis.py` — `component`,
`correct`, `total`, `accuracy`, `complete`, plus runtime statistics. The denominator is
always the **subset**, never the number of items that happened to answer or get graded:
an item that errored or was never judged is not excused from the denominator, it is
simply not correct.

## Resuming

`run` with no `--results-dir` mints a new directory. `run --results-dir <existing>`
resumes into it.

> An id counts as **done** only if its row has no `error` **and** a non-empty
> `response`. Anything else is pending and gets retried.

This is the fix for the bug that shaped this repo. The previous implementation resumed
on id-presence alone, so a row written with `error != null` and an empty answer counted
as complete and was never retried — rows had to be deleted by hand from a 21 MB JSONL
file to get those items to run. A resume now rewrites `responses.jsonl` once, before any
worker starts, keeping the good rows and banking the rest in `failures.jsonl`.

A resume also compares `endpoint`, `model`, `mode` and subset size against `config.json`
and **aborts if they changed**, rather than blending two endpoints into one result set.
`--force` proceeds and appends the change to `config_history`, so the artifact records
the mixing instead of hiding it.

## Grading

The two LLM-judged components do not grade themselves. The clients only produce
answers; grading is a separate step driven by a skill:

- `.claude/skills/judge-hle/` — the HLE equality-checker rubric
- `.claude/skills/judge-aalcr/` — the long-context rubric

Both follow the same procedure: `pending` lists ungraded ids, **one isolated subagent
grades each id**, `record` writes its verdict, `collect` folds them into `score.json`.
The isolation is the point — a grader that has seen another item's gold answer is
contaminated, so no agent ever sees more than the single question it is grading.

`record` refuses an id that is not in the run's subset. An earlier grading wave, running
while a container was down, wrote a docker error string as an item id; it would have
counted as a spurious incorrect had it not been spotted by hand.

## Reference numbers

`tools/import_legacy.py` converts the original `eai-v1.0` run into four dated result
directories, which is also the regression test for this repo — the numbers must come
back unchanged:

| Component | Result |
|---|---|
| BFCL-500 | 373/500 = 74.60% |
| HLE-250 with tools | 114/250 = 45.60% |
| AA-LCR-25 | 20/25 = 80.00% |
| **Composite** | **66.73 / 100** |
| HLE-250 no tools (reference) | 89/250 = 35.60% |

Those were measured against GLM-5.3 on 8×B200 under sglang. The no-tools pass is
depressed by a since-removed generation cap: 36 of 250 items truncated to empty answers
and all 36 scored zero. GLM-5.3's published 62.5 is **HLE with tools** — never compare
it against the no-tools figure, and note that 45.60% uses a keyless DuckDuckGo
`web_search` and a single repeat, so it is not like-for-like either.

`tools/selftest.py` checks the resume rule, the config guard and the grading round-trip
offline, without an endpoint. Run it after touching `resultdir.py`.

## Notes

`endpoint.py` and `resultdir.py` are **vendored identically** into each client rather
than hoisted into a shared package. The clients are meant to be separately copyable, so
the duplication is deliberate; keep the copies in sync by hand when changing either.
