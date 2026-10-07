# HLE-250

250 questions from Humanity's Last Exam (`cais/hle`), in either of two modes.

| Mode | Flag | Generation cap | Tools |
|---|---|---|---|
| with tools (default) | `--tools` | 131,072 per request | `python` |
| no tools | `--no-tools` | 131,072 | none |

**Both modes select the identical 250 ids**, so they are directly comparable;
`config.json` records `"mode"` as the discriminator. The subset is chosen without an
RNG — sort the text-only rows by id, take an even stride — so any machine picks the same
250. Image questions are excluded outright and therefore never enter the denominator.

With tools is the headline setting: it is what the official methodology and the model
cards report. The no-tools pass is additionally depressed by its cap.

## Running

```bash
python3 hle/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3 --tools
python3 hle/src/main.py run --results-dir hle/results/<ts>-tools     # resume
python3 hle/src/main.py run --endpoint ... --model ... --limit 5     # smoke test
```

`--concurrency` defaults to 256 — high on purpose, because most of an item's wall-clock
is local tool execution rather than endpoint time, so in-flight requests stay well under
that. The Python worker pool (16 workers) is the real ceiling on tool throughput; raise
it with `HLE_PY_WORKERS` if tool calls queue.

`--max-conc-context N` (tools mode, off by default) caps the summed context of the
running items at N tokens. Concurrency counts items, but a long tool loop grows to
hundreds of thousands of tokens, and once the conversations in flight outgrow the
server's KV pool the prefix cache thrashes: every round re-prefills its whole
conversation and every item slows at once. Under the gate each item holds its current
context size and applies again before every request; one that no longer fits pauses
until running items finish or pause in turn. The oldest item goes first, so as contexts
build up the youngest items pause and the effective concurrency falls, rising again as
the long items finish. An item that alone exceeds N runs by itself. The hold is the
context an item brings to a request, not what the request generates, so set N below the
KV pool with headroom for output. Rows record `ctx_pause_s`, the time an item spent
paused after it started, and a `[ctx gate]` line reports the gate at most once a minute
while items wait.

Then grade with the **judge-hle** skill, which ends by
running `collect`. `audit --results-dir DIR` reports integrity (nothing marked correct
with an empty response, ids match the subset) and scans the stored tool traces for hits
on dataset-hosting domains — the python tool has internet access, so the model can in
principle retrieve the question set, and that exposure is worth quantifying rather than
assuming absent.

## Dataset access

`cais/hle` is **gated**. Either export `HF_TOKEN` from an account that accepted the
terms, or pre-populate the HF cache and run with `HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1`. The client checks for one of those up front rather than failing with
a 401 several minutes into a run.

## The agentic loop

With tools, the model is driven until it stops calling them or has used its **tool-call
budget of 512 calls** (`--max-tool-calls N`; 0 = uncapped). Every tool result ends with
the budget line — `[tool budget: 37 of 512 tool calls used, 475 remaining]` — so the
model always knows what it has left. Calls past the budget are answered "Not executed"
instead of running, and the item goes to the forced final below; `tool_budget_hit` is
recorded on the row. There is no round cap and no total generation budget — each request
carries the suite-wide `max_tokens` of 131,072, and beyond that the ceilings are the tool
budget, the server's context window and the model deciding it is done. Every tool round
executes at least one call, so the budget also bounds an item to 512 tool rounds.

The tool budget exists because the uncapped loop has no other end: one fp4 GLM-5.3 item
looped for about 9,000 rounds until it was stopped by hand. It was 1024 until October
2026, when Kimi-K3 runs repeated one identical call hundreds of times (828 runs of the
same python code, one search query 972 times) and grew 450k–920k-token contexts that
starved every other in-flight item of KV cache. (The reference run sent no `max_tokens`
at all. A server that refuses prompt + `max_tokens` over the window now reports the
context full about 131k tokens earlier, which goes to the forced final below.)
`--gen-budget N` caps the
completion tokens summed over the loop; `--max-tokens 0` drops the per-request cap.

The round cap and generation budget are off by default because they silently produced empty
answers rather than shorter ones: under an earlier 40-round cap, all 15 items that hit it returned nothing at all,
because the code took the last assistant turn and on a tool-calling turn that is empty.
An item must never end without an answer merely because it ran long, so:

- if the loop ends with no answer, a **forced final** step appends "answer now" with
  `tool_choice="none"`;
- if the context is full so that is impossible, it rebuilds a short conversation from
  the question plus a digest of the tool findings and asks there.

`forced_final` and `context_full` are recorded per row, along with `prompt_tokens`,
`final_context_tokens` and `interaction_tokens` = final context − prompt. The final
context is the last request of the item's own conversation (a rebuilt forced-final
conversation does not count), so interaction is every assistant turn and tool result the
item added. `score.json` reports its median, p90 and max.

## The python tool

Model-written code runs unsupervised in a throwaway directory with a hard **8 GB address
space** and **256 MB file size** limit, applied by a wrapper script inside the child
rather than `preexec_fn` (which is not thread-safe). One unbounded allocation once
OOM-killed the whole container.

BLAS threads are pinned to 4. `RLIMIT_AS` caps address space rather than resident
memory, and OpenBLAS reserves a stack per thread — across 64 threads that overruns any
sane ceiling before a single array is allocated, which is why a first attempt at a 4 GB
limit broke numpy outright.

There is no search tool. Runs before October 2026 also offered `web_search`
(DuckDuckGo via `ddgs`, Wikipedia API fallback); it was removed because from the cluster
a third to a half of its calls failed outright, many of the rest returned unrelated pages,
and models retried the same query hundreds of times against it. Published with-tools
numbers generally include search, a caveat when comparing against them.

## Files

`main.py` CLI · `dataset.py` subset selection and prompts · `runner.py` the two
generation modes · `ctxgate.py` the context budget over running items · `tools.py`
the python sandbox · `endpoint.py` async client
(vendored) · `resultdir.py` results-directory contract (vendored).
