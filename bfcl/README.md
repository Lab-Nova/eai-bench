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

`run --benchmark` stops using the suite to score an endpoint and starts using it to time
one. Three things are fixed, and each is refused rather than adjusted, because a knob
you have to go and check is not a benchmark:

1. **The tasks.** The 500-task subset minus the five in `subset.LONG_TAIL` — 495 tasks.
2. **The order.** Written to `dispatch_order.json` and fingerprinted into
   `config.json` before the first request goes out, and each run compares its
   fingerprint with the previous benchmark run's and says whether they match. The hash
   is over the *ordered* id list, so one number pins both the tasks and their order —
   two runs can be shown to have been the same workload rather than assumed to have been.
3. **The concurrency.** 256, always. `--concurrency` anything else is an error.

One thing follows from wanting the same workload every time: `--results-dir` is refused.
A resume generates only the ids still missing from the result files, which is a smaller
workload in a different order; a half-finished benchmark run is discarded, not continued.

The repair pass is **not** one of the things benchmark mode turns off. A row holding
`Error during inference` is a request the gateway dropped, not a shorter workload, and
leaving it there would put a wrong answer in the score and a missing request in the
timing — so `--max-attempts` is 3 in every mode, and the pass re-runs the same ids in
the same relative order. It does cost wall clock, so a run that needed one says so, and
two runs are comparable on wall clock only if both finished in a single pass.
`config.json` records `attempts_used` alongside `generate_wall_s`.

The run prints its wall clock, request-seconds, slot utilisation and latency
percentiles.

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

### The long tail, and how it was found

`main.py latency` ranks every task by the per-request latency bfcl-eval already records
in its result rows — summed over every turn and step, because one task holds one worker
thread from its first request to its last. It reads finished runs off disk, so it needs
no endpoint and no venv, and it aggregates across runs on purpose: in a single run the
ranking is as much a picture of the endpoint's bad minutes as of the suite. One task
here moved from 5 s to 1088 s between two runs.

Over eight runs, against a subset median of ~6 s:

```
      med      min      max  runs fail  steps  out_tok  task
    662.3    602.9   2215.5     3    3     16    72505  multi_turn_miss_func_147
    499.1    211.0    818.6     5    2     11    22930  multi_turn_long_context_171
    478.8      5.5   1087.9     6    0      1    18494  simple_java_74
    324.2    118.4   1466.4     6    0     11    25854  multi_turn_base_171
    321.5    150.2   1093.3     7    0     14    44909  multi_turn_miss_param_171
    283.1     54.5    681.4     6    0     15    10930  multi_turn_miss_func_66
```

That top five is about a fifth of all request-seconds. Three of them are one scenario,
index 171, which the suite repeats once per multi_turn category: a three-turn
travel-booking-and-messaging session, 10-15 model calls. `simple_java_74` is the odd one
— a single-turn question about serialising an XML surrogate pair on which the model runs
away to tens of thousands of output tokens instead of emitting the one call it is asked
for.

The `fail` column is why `multi_turn_miss_func_147` is on the list. A failed row carries
no latency and drops out of the median, which quietly flatters exactly the tasks slow
enough to hit a gateway timeout: the worst task in the suite is missing from half the
runs for that reason, and a ranking that reads only completed rows discounts the task
most worth removing.

It is a shoulder, not a cliff — the sixth is at 283 s. `latency` prints a DRIFT warning
when the measured top five stops matching `LONG_TAIL`, and stops there: editing the list
changes the workload, so it is a deliberate edit to `subset.py`, never something a
measurement command does on its own.

### It is not this endpoint's BFCL score

495 tasks is a different denominator, and the five that go are not a random five —
they are the hardest multi-turn sessions in the sample, so the pass rate drifts up. The
run records `benchmark: true` and `long_tail_excluded` in both `config.json` and
`score.json`, `collect` says so in as many words, and `synthesis.py` skips any run
carrying the flag when it looks for the latest BFCL number.

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
