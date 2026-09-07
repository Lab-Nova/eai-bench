# BFCL-500

500 tasks from the Berkeley Function Calling Leaderboard v4, run through the
third-party `bfcl-eval` package and graded by machine (AST + execution state). There is
no judge skill: `run` then `collect` produces `score.json` directly.

The subset is proportional to each category's share of the suite (largest-remainder
apportionment, every category guaranteed at least one task) and picked by an even stride
over sorted ids — no RNG. `web_search` and `memory_*` are excluded: they need a SerpAPI
key and a vector backend respectively.

## Running

```bash
python3 bfcl/src/main.py setup                                  # once: bfcl/.venv, ~5 GB
python3 bfcl/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3
python3 bfcl/src/main.py collect --results-dir bfcl/results/<ts>
python3 bfcl/src/main.py status  --results-dir bfcl/results/<ts>

python3 bfcl/src/main.py run --endpoint … --model … --benchmark  # fixed timing workload
python3 bfcl/src/main.py latency                                 # rank tasks by latency
```

`--model` is the server's `--served-model-name`. `run` re-executes itself under the venv
interpreter automatically. Re-running `run` against an existing `--results-dir` resumes:
`bfcl generate` skips ids already present in its result files.

`setup` installs `soundfile` alongside the pin. `bfcl-eval` 2026.3.23 does not declare it,
but touching `bfcl_eval.constants.model_config` imports the entire handler registry, and
the Qwen handler reaches `qwen_agent` -> `import soundfile`. Without it a clean install
cannot register a model at all — `run` dies on `ModuleNotFoundError: No module named
'soundfile'` before it sends a single request. If a future pin adds more undeclared
imports, they belong in `BFCL_EXTRA_DEPS` next to this one.

## Benchmark mode

`run --benchmark` runs the full deterministic **500-task subset**, in the recorded
fixed dispatch order, at concurrency **256**, with a **300-second generation deadline**.
The deadline starts before the first generation pass and covers imports, dispatch,
all model calls, and any repair passes. Server startup and grading are outside it.
`--target` other than 500, a different concurrency, and `--results-dir` are refused.

At the deadline the generation process group is stopped. Complete result rows are
preserved; unfinished tasks receive `status: "timeout"` and an inference-error result.
Timeouts are terminal for this run and count as incorrect in the **500-task denominator**.
`score.json` reports `timed_out`, `timeout_ids`, and `responded` separately.
`benchmark_complete` means every task was graded and either answered or timed out;
`complete` still requires every task to have an answer without an inference error.
No repair pass starts after the deadline. Before it, the usual maximum of three
passes applies to inference errors. Ordinary BFCL mode has no new time cap.

`config.json` records `generation_timeout_s`, `generate_wall_s`, `attempts_used`,
and the workload fingerprint. Timeout cleanup can add a small scheduling overhead
to the reported wall time. Client cancellation closes requests; a shared server may
finish cancellation asynchronously. The baseline harness stops its dedicated server
after collection.

### The order was already fixed; it was just never written down

bfcl-eval sorts every test case by `(single-turn before multi-turn, category, index)`
and both seeds and refills its thread pool from a heap of that key, so submission order
never depended on timing. `subset.dispatch_order` recomputes it from the ids alone.
Two consequences worth knowing:

* Every multi_turn task sits in the last fifth of the queue. With 256 threads and 412
  single-turn tasks ahead of them, the expensive ones start last, so the wall clock is
  roughly `(single-turn wave) + (slowest multi-turn task)` — which is why removing five
  tasks moves it as much as it does.
* A **resumed** run has a different, shorter queue: `collect_test_cases` drops ids
  already present in the result files. Same tasks, different workload. Hence the refusal.

### Historical exclusion-based runs

Earlier benchmark revisions excluded five tasks (495 total), then ten (490 total).
Those policies have been replaced by the full 500-task, five-minute deadline above.
Their saved fingerprints, timings, and denominators remain historical results and
must not be compared as identical workloads. The old `LONG_TAIL` list is retained
only as reference for latency analysis; benchmark mode no longer applies it.

Benchmark scores carry `benchmark: true` and are excluded from `synthesis.py`:
a deadline-limited score is separate from an uncapped accuracy run.

## Five things that silently produce a wrong number

Each of these was found by reconstruction, and each fails quietly rather than loudly.

1. **`--run-ids` is mandatory.** Without it the id file is ignored entirely and the full
   ~4,400-task suite runs. The flag's own help text says it runs the listed ids *in
   addition to* `--test-category`; the code does the opposite and replaces it.

2. **`--partial-eval` is mandatory.** `evaluate` raises outright when the number of
   result rows differs from the number of prompt entries, which a 500-of-4400 subset
   guarantees.

3. **The endpoint must go in `$BFCL_PROJECT_ROOT/.env`, not the environment.** Both
   subcommands call `load_dotenv(..., override=True)`, so that file *beats* an exported
   `OPENAI_BASE_URL`. Exporting the endpoint looks like it works and silently does not
   the moment a stale `.env` exists. `OPENAI_API_KEY` must be non-empty (the openai
   client constructor raises otherwise) even though a local server ignores it.

4. **The registry name may contain no `_` and no `/`.** `base_handler` does
   `registry_name.replace("/", "_")` to name the result and score directories, and
   `eval_runner` reverses it with `replace("_", "/")` to turn that directory name back
   into a lookup key. A name with an underscore fails to find itself at evaluation time.

5. **`underscore_to_dot=True` is not optional.** BFCL rewrites dotted tool names
   (`ChaDri.change_drink` → `ChaDri_change_drink`) for OpenAI-style handlers and only
   undoes it when this flag is set. It is grading-side only, so flipping it needs no
   regeneration — but with it wrong, whole categories score zero. Two smoke categories
   went 0% → 100% and 50% → 100% when it was corrected.

`--skip-server-setup` is the flag everyone reaches for here and it is the wrong one: it
is only read for `OSSHandler` models, and this registration uses the API handler, so it
and its `LOCAL_SERVER_*` variables are never consulted.

## Registration, and why it is in-process

`bfcl-eval` has no plugin mechanism — `MODEL_CONFIG_MAPPING` is three static dict
literals, and the upstream instruction for adding a model is to edit that source file.
The previous incarnation did exactly that, appending a `ModelConfig` to
`site-packages/bfcl_eval/constants/model_config.py`, which does not survive a reinstall
and leaves no record of what was registered.

`shim.py` mutates the mapping in-process instead. That works because every consumer does
`from ... import MODEL_CONFIG_MAPPING`, binding the same dict object, and every lookup
happens at call time. `BFCL_PROJECT_ROOT` must be set **before** any `bfcl_eval` import,
because that package creates `result/`, `score/` and `.file_locks/` at import time.

## Resume, and the error rows

`bfcl generate` catches every inference exception and writes `"Error during inference:
<e>"` into the result field **under a valid id**. On the next run that id counts as
generated, is skipped forever, and `evaluate` grades it wrong. Upstream's reasoning —
retrying will not help at temperature 0.001 — holds for a malformed model response and
not at all for a connection error from a server restart.

So `run` scans for those rows after each generate and re-runs exactly them, up to
`--max-attempts` (default 3). The retry uses `--run-ids --allow-overwrite`, the one
combination that leaves the existing result file in place while regenerating the listed
ids; the write is an id-keyed upsert, so no duplicates appear. The narrowed id file is
scratch — `subset.json` is the source of truth and the full list is restored afterwards.

## Isolation

Each run gets its own `BFCL_PROJECT_ROOT` at `<results-dir>/bfcl_root`. This is
mandatory, not stylistic: `evaluate` folds **every** score file it finds in its score
directory into the leaderboard CSVs, so two runs sharing a root blend into each other's
numbers.

## The score

`score.json` reports a **flat pass rate** — total correct over the 500-task subset — so
BFCL sits on the same footing as the other two components. bfcl-eval's own
category-weighted "Overall Acc" is carried as a labelled footnote only: it counts the
categories we did not run as N/A and is not comparable across components. In the
reference run the flat rate is 74.60% and the weighted figure 40.60%.

`--num-threads` defaults to 256. Note that bfcl-eval's own default is **1** for an API
handler — only OSS handlers get a concurrent default — so leaving it unset runs the whole
subset serially. Sampling uses bfcl-eval's default temperature 0.001, which differs from
the other two components (temperature 1.0 / top_p 0.95 from the model card); the value is
recorded in `config.json` so the difference is visible rather than assumed.

## Files

`main.py` CLI and orchestration · `shim.py` in-process registration and `.env` ·
`subset.py` the deterministic 500-task subset, `LONG_TAIL` and the dispatch order ·
`resultdir.py` results-directory contract (vendored).
