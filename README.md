# eai-bench — Endpoint Accuracy Index

Three benchmark clients that measure one OpenAI-compatible endpoint, plus a synthesis
step that combines them into a single number.

| Component | Size | Grading | What it measures |
|---|---|---|---|
| `bfcl/` | 500 tasks | machine (`bfcl-eval`, AST + state) | function / tool calling |
| `hle/` | 250 questions | LLM judge | frontier reasoning (Humanity's Last Exam) |
| `aalcr/` | 100 questions × 5 (avg@5) | LLM judge (dataset's v1.1 prompts) | long-context reasoning (AA-LCR) |

The index is the **equally-weighted mean of the three accuracies**. Each client is
self-contained: it takes `--endpoint` and `--model`, writes one timestamped results
directory, and resumes into that directory if interrupted.

Two long-context probes sit alongside the index. They are **not** averaged into it:

| Probe | Size | Grading | What it measures |
|---|---|---|---|
| `counting_stars/` | 2 langs × 15 lengths (64k–960k) × 10 samples (avg@10) | machine (needle scores) | multi-needle recall with corrections, EN + ZH |
| `babilong_qa3/` | 4 lengths (0k/128k/256k/384k) × 100 | machine (BABILong match) | three-fact reasoning over a long haystack |

They exist because they move when the serving stack changes. Weight and KV-cache
quantizations that leave the three index components flat can still cost qa3 accuracy at
long context. `synthesis.py` prints them in a separate section and leaves the composite
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

Sampling is temperature 0.6 / top_p 0.95 in hle, aalcr, counting_stars and babilong_qa3
(`endpoint.DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P`); bfcl keeps bfcl-eval's own 0.001.
Results from before 2026-10-05 used temperature 1.0 / top_p 0.95; each run's
`config.json` records its sampling.

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

A reasoning model can also finish cleanly with nothing but reasoning: it runs to
`max_tokens` with empty content. That is the model's answer, not an error, and
resampling it would inflate the score. So `endpoint.finish_row` writes
`NO ANSWER: output limit exceeded before a final answer was produced.` (or
`NO ANSWER: the model returned an empty response.` below the cap) as the response and
sets `no_answer` to `output_limit` / `empty`. The row counts as done, a resume keeps it,
and every grader scores it wrong.

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

GLM-5.3 BF16 weights with a BF16 KV cache, under sglang on 4 GB300 trays (16 GPUs, TP16),
October 2026, at the 131,072-token cap. HLE, AA-LCR, Counting-Stars and qa3 sample at
temperature 0.6 / top_p 0.95 (run of 2026-10-05). BFCL samples at bfcl-eval's own 0.001 and
is the run of 2026-10-01. Accuracy SE is binomial over the subset, except AA-LCR (over the
100 per-question means of avg@5) and Counting-Stars (over its 30 language x length cells).
Median-interaction SE is a 2000-resample bootstrap over the items that produced an answer
(`compare/eai_stats.py` in workspace-needle).

| Component | Accuracy | Median interaction (tokens) |
|---|---|---|
| BFCL-500 | 374/500 = 74.80 ± 1.94% | 195 ± 7 |
| HLE-250 with tools | pending | |
| AA-LCR-100 (v1.1), avg@5 | 365/500 = 73.00 ± 3.67% | 1,727 ± 132 (n=456) |
| **Composite** | pending (needs HLE) | |
| Counting-Stars (probe), avg@10 | pending | |
| BABILong qa3 (probe) | pending | |

On AA-LCR, 44 of the 500 samples (8.8%) reasoned until the cap without a final answer.
They are written as `NO ANSWER: output limit exceeded ...` (see Resuming) and count as
wrong. At temperature 1.0 the same server ran past the cap on 1 of 100 questions.

`tools/import_legacy.py` converts the original `eai-v1.0` run into four dated result
directories, which is also the regression test for this repo: the imported scores must
come back unchanged.

`tools/selftest.py` checks the resume rule, the config guard and the grading round-trip
offline, without an endpoint. Run it after touching `resultdir.py`.

## Notes

`endpoint.py` and `resultdir.py` are **vendored identically** into each client rather
than hoisted into a shared package. The clients are meant to be separately copyable, so
the duplication is deliberate; keep the copies in sync by hand when changing either.
