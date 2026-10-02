# eai-bench — Endpoint Accuracy Index

Three benchmark clients that measure one OpenAI-compatible endpoint, plus a synthesis
step that combines them into a single number.

| Component | Size | Grading | What it measures |
|---|---|---|---|
| `bfcl/` | 500 tasks | machine (`bfcl-eval`, AST + state) | function / tool calling |
| `hle/` | 250 questions | LLM judge | frontier reasoning (Humanity's Last Exam) |
| `aalcr/` | 100 questions | LLM judge (dataset's v1.1 prompts) | long-context reasoning (AA-LCR) |

The index is the **equally-weighted mean of the three accuracies**. Each client is
self-contained: it takes `--endpoint` and `--model`, writes one timestamped results
directory, and resumes into that directory if interrupted.

Two long-context probes sit alongside the index. They are **not** averaged into it:

| Probe | Size | Grading | What it measures |
|---|---|---|---|
| `counting_stars/` | 2 langs × 15 lengths (64k–960k) × 5 samples | machine (needle scores) | multi-needle recall with corrections, EN + ZH |
| `babilong_qa3/` | 4 lengths (0k/128k/256k/384k) × 100 | machine (BABILong match) | three-fact reasoning over a long haystack |

They exist because they move when the serving stack changes. Weight and KV-cache
quantizations that leave the three index components flat can still cost 10–15 points on
qa3 at 256k. `synthesis.py` prints them in a separate section and leaves the composite
alone, so it stays comparable with the reference numbers below.

```
eai-bench/
  README.md
  synthesis.py            latest-or-specified results per client -> composite
  hle/  aalcr/  bfcl/     one benchmark client each: src/, results/, README.md
  counting_stars/  babilong_qa3/   the long-context probes, same layout
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

python3 counting_stars/src/main.py run --endpoint $EP --model $MODEL   # probes: graded
python3 babilong_qa3/src/main.py   run --endpoint $EP --model $MODEL   # as they finish

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
`correct`, `total`, `accuracy`, `complete`, plus runtime statistics.

Every component asks for at most **131,072 tokens per request** (`--max-tokens`), and
every `score.json` reports `interaction_tokens_median` (with p90 and max): final context
length − prompt length per item. For a single-turn item that is its completion; for a
BFCL multi-turn task or an HLE tool loop it is everything the conversation added between
the first prompt and the end of the last response. `synthesis.py` prints the median on
each component's runtime line. The denominator is
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

### GLM-5.3 BF16, current suite

GLM-5.3 BF16 weights with a BF16 KV cache, under sglang on 4 GB300 trays (16 GPUs, TP16),
October 2026. Every component ran in full at the 131,072-token cap. Accuracy SE is
binomial over the whole subset; Counting-Stars SE is over its 30 (language, length)
cells. Median-interaction SE is a 2000-resample bootstrap over the items that have a
count (`compare/eai_stats.py` in workspace-needle).

| Component | Accuracy | Median interaction (tokens) |
|---|---|---|
| BFCL-500 | 374/500 = 74.80 ± 1.94% | 195 ± 7 |
| HLE-250 with tools | 131/250 = 52.40 ± 3.16% | 22,839 ± 2,764 |
| AA-LCR-100 (v1.1) | 78/100 = 78.00 ± 4.14% | 2,198 ± 462 (n=99) |
| **Composite** | **68.40 / 100** | |
| Counting-Stars (probe) | 137.53/150 = 91.69 ± 1.21% | 1,948 ± 92 |
| BABILong qa3 (probe) | 220/400 = 55.00 ± 2.49% | 22,316 ± 2,114 (n=373) |

The qa3 split by length is 0k 100/100, 128k 51/100, 256k 40/100 and 384k 29/100. qa3
uses the clarified question (see `babilong_qa3/README.md`), so it is not comparable with
the 256k reference rows measured with the upstream question.

Answers that hit the cap are empty and count as wrong. They stay in the accuracy
denominator but have no interaction count: 1 AA-LCR item and 27 qa3 items. Errors and
dropped streams from server load were retried until every item had a response. BFCL
multi-turn retries start from clean environments (see `bfcl/README.md`).

The same suite against two quantized serving stacks, each on 2 trays (8 GPUs):

| Component | MXFP4 (MR-GPTQ v2) / BF16 KV | MXFP4 (MR-GPTQ v2) / FP8 KV (FlashMLA, per-128 scaled) |
|---|---|---|
| BFCL-500 | 77.00 ± 1.88% · 198 ± 10 | 75.00 ± 1.94% · 198 ± 11 |
| HLE-250 with tools | 47.20 ± 3.16% · 16,874 ± 2,213 | 54.80 ± 3.15% · 19,946 ± 4,047 |
| AA-LCR-100 | 80.00 ± 4.00% · 2,544 ± 477 | 80.00 ± 4.00% · 2,478 ± 411 |
| **Composite** | **68.07** | **69.93** |
| Counting-Stars | 91.15 ± 1.47% · 2,072 ± 158 | 91.25 ± 1.25% · 1,911 ± 101 |
| BABILong qa3 | 56.50 ± 2.48% · 20,921 ± 2,360 | 60.00 ± 2.45% · 21,763 ± 2,748 |

Each cell is accuracy · median interaction tokens. On the MXFP4 / BF16 KV run, one HLE
item went into a runaway tool loop: about 9,000 rounds, with the context growing from
81k to 272k tokens. It was stopped by hand and scored wrong, so that run's HLE is
`complete: false`. The BF16 run solved the same item in 37 rounds. One Counting-Stars
answer on that run (EN-640k) hit the cap. No difference from BF16 exceeds 1.5 combined
standard errors.

### Legacy eai-v1.0 run

`tools/import_legacy.py` converts the original `eai-v1.0` run into four dated result
directories, which is also the regression test for this repo — the numbers must come
back unchanged:

| Component | Result |
|---|---|
| BFCL-500 | 373/500 = 74.60% |
| HLE-250 with tools | 114/250 = 45.60% |
| AA-LCR-25 (legacy 25-question subset, v1.0 keys) | 20/25 = 80.00% |
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
