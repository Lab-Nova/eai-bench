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

Every component asks for at most **131,072 tokens per request** (`--max-tokens`).
*Interaction* is final context length − prompt length per item. For a single-turn item
that is its completion; for a BFCL multi-turn task or an HLE tool loop it is everything
the conversation added between the first prompt and the end of the last response.
Every `score.json` reports:

- `interaction_tokens_median` with a 2000-resample bootstrap `_median_se`, plus `_mean`,
  `_p90`, `_max` and `_n`, over the items that **produced an answer**. Runaways (below)
  sit at the cap and would swamp the mean, so they are left out of all of these.
- `no_answer` (`output_limit` / `empty` counts) and `output_limit_rate`: the share of
  error-free items that reasoned into the token cap or the context window without a final
  answer. These are the runaways. They score wrong, and their rate is reported on its own.
  Runs written before the NO ANSWER marker are read the same way. bfcl-eval's rows do not
  record a cap hit, so BFCL has no `no_answer`.
- `interaction_by`, the same numbers per stratum, where one pooled median would describe
  only the majority: BFCL `single_turn` / `multi_turn` (82% of the tasks are single-turn,
  at about 1/25 the interaction) and qa3 per haystack length.
- HLE also reports tool `rounds_median`, `_mean`, `_p90` and `_max` over answered items.

`synthesis.py` prints the median, the strata and the output-limit rate on each
component's runtime line. The accuracy denominator is
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

### GLM-5.3

GLM-5.3 BF16 weights with a BF16 KV cache, under sglang on 4 GB300 trays (16 GPUs, TP16),
October 2026, at the 131,072-token cap. HLE, AA-LCR, Counting-Stars and qa3 sample at
temperature 0.6 / top_p 0.95 (run of 2026-10-05). BFCL samples at bfcl-eval's own 0.001 and
is the run of 2026-10-01. Accuracy SE is binomial over the subset, except AA-LCR (over the
100 per-question means of avg@5) and Counting-Stars (over its 30 language x length cells).
Interaction columns are over answered items only (median ± bootstrap SE, and mean); the
output-limit column counts the runaways they leave out (see above).

| Component | Accuracy | Interaction median | Interaction mean | Output limit |
|---|---|---|---|---|
| BFCL-500 | 374/500 = 74.80 ± 1.94% | 195 ± 7 | 1,631 | n/a |
| · single-turn (412) | | 160 ± 10 | 260 | |
| · multi-turn (88) | | 3,864 ± 412 | 8,052 | |
| HLE-250 with tools | pending | | | |
| AA-LCR-100 (v1.1), avg@5 | 365/500 = 73.00 ± 3.67% | 1,727 ± 135 | 3,595 | 44/500 = 8.8% |
| **Composite** | pending (needs HLE) | | | |
| Counting-Stars (probe), avg@10 | 88.74 ± 1.93% | 1,719 ± 60 | 2,273 | 7/300 = 2.3% |
| BABILong qa3 (probe) | 201/400 = 50.25 ± 2.50% | 9,884 ± 2,073 | 21,383 | 62/400 = 15.5% |
| · 0k | 100/100 | 634 ± 60 | 890 | 0/100 |
| · 128k | 50/100 | 23,284 ± 5,789 | 28,110 | 15/100 |
| · 256k | 30/100 | 26,689 ± 5,859 | 33,853 | 19/100 |
| · 384k | 21/100 | 25,194 ± 4,543 | 27,877 | 28/100 |

At temperature 1.0 the same AA-LCR server ran past the cap on 1 of 100 questions; at 0.6
the rate is 8.8%.

### Kimi-K3

Kimi-K3 as released (MXFP4 routed experts, BF16 elsewhere) with a BF16 KV cache, under
sglang v0.5.21 on 2 GB300 trays (8 GPUs, TP8 + DCP8, DSPARK speculative decoding, the
SGLang cookbook recipe; KV pool 995,840 tokens). The runs date from October 2026, at the
131,072-token cap and the sampling above. BFCL, AA-LCR and the probes ran at 79a5713.
HLE ran at 78b7895: python only, with each turn's reasoning sent back. The interaction
columns are recomputed with the current code. glm-5.3 gave no verdict on 3 HLE items, so
Sonnet graded those (1 correct).

| Component | Accuracy | Interaction median | Interaction mean | Output limit |
|---|---|---|---|---|
| BFCL-500 | 384/500 = 76.80 ± 1.89% | 272 ± 11 | 1,424 | n/a |
| · single-turn (412) | | 230 ± 10 | 386 | |
| · multi-turn (88) | | 4,046 ± 282 | 6,285 | |
| HLE-250 with tools | 135/250 = 54.00 ± 3.15% | 14,743 ± 1,565 | 29,562 | 0/250 |
| AA-LCR-100 (v1.1), avg@5 | 433/500 = 86.60 ± 3.11% | 626 ± 20 | 1,165 | 0/500 |
| **Composite** | **72.47** | | | |
| Counting-Stars (probe), avg@10 | 89.31 ± 2.64% | 3,141 ± 126 | 6,912 | 12/300 = 4.0%, + 5 below |
| BABILong qa3 (probe) | 250/400 = 62.50 ± 2.42% | 4,270 ± 820 | 10,157 | 56/400 = 14.0% |
| · 0k | 100/100 | 1,236 ± 121 | 1,396 | 0/100 |
| · 128k | 82/100 | 7,370 ± 1,608 | 8,285 | 6/100 |
| · 256k | 40/100 | 13,814 ± 2,060 | 18,084 | 19/100 |
| · 384k | 28/100 | 12,112 ± 2,954 | 16,098 | 31/100 |

Counting-Stars scores EN 0.820 and ZH 0.967. Its lengths are counted in GLM-5.3 tokens. In
K3 tokens the EN-960k prompt is 962,719, so the `max_tokens` the client derives from the
GLM-5.3 count overran the window and the server returned HTTP 400. Those 10 samples were
resumed with `--max-tokens context` (85,838). With that prompt, though, the KV pool has
room for only about 32k output tokens. 5 of the 10 stopped at 32,255 when sglang reported
the pool full. They have no answer and score 0 (`no_answer` records them as `empty`). The
cause is the serving setup, not the model, and it costs at most 1.7 points.

`tools/import_legacy.py` converts the original `eai-v1.0` run into four dated result
directories, which is also the regression test for this repo: the imported scores must
come back unchanged.

`tools/selftest.py` checks the resume rule, the config guard and the grading round-trip
offline, without an endpoint. Run it after touching `resultdir.py`.

## Notes

`endpoint.py` and `resultdir.py` are **vendored identically** into each client rather
than hoisted into a shared package. The clients are meant to be separately copyable, so
the duplication is deliberate; keep the copies in sync by hand when changing either.
