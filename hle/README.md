# HLE-250

250 questions from Humanity's Last Exam (`cais/hle`), in either of two modes.

| Mode | Flag | Generation cap | Tools |
|---|---|---|---|
| with tools (default) | `--tools` | none — `max_tokens` is not sent | `python`, `web_search` |
| no tools | `--no-tools` | 65,536 | none |

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
that. The tool pools (16 Python workers, 8 search) are the real ceiling on tool
throughput; raise them with `HLE_PY_WORKERS` / `HLE_SEARCH_WORKERS` if tool calls queue.
Then grade with the **judge-hle** skill, which ends by
running `collect`. `audit --results-dir DIR` reports integrity (nothing marked correct
with an empty response, ids match the subset) and scans the stored tool traces for hits
on dataset-hosting domains — with web search enabled the model can in principle retrieve
the question set, and that exposure is worth quantifying rather than assuming absent.

## Dataset access

`cais/hle` is **gated**. Either export `HF_TOKEN` from an account that accepted the
terms, or pre-populate the HF cache and run with `HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1`. The client checks for one of those up front rather than failing with
a 401 several minutes into a run.

## The agentic loop

With tools, the model is driven until it stops calling them. There is **no round cap and
no `max_tokens`** — the only ceilings are the server's context window and the model
deciding it is done. Items legitimately run past 100 rounds; the deepest observed in the
reference run was 491 rounds.

Both caps were removed because they silently produced empty answers rather than shorter
ones: under an earlier 40-round cap, all 15 items that hit it returned nothing at all,
because the code took the last assistant turn and on a tool-calling turn that is empty.
An item must never end without an answer merely because it ran long, so:

- if the loop ends with no answer, a **forced final** step appends "answer now" with
  `tool_choice="none"`;
- if the context is full so that is impossible, it rebuilds a short conversation from
  the question plus a digest of the tool findings and asks there.

`forced_final` and `context_full` are recorded per row.

## The python tool

Model-written code runs unsupervised in a throwaway directory with a hard **8 GB address
space** and **256 MB file size** limit, applied by a wrapper script inside the child
rather than `preexec_fn` (which is not thread-safe). One unbounded allocation once
OOM-killed the whole container.

BLAS threads are pinned to 4. `RLIMIT_AS` caps address space rather than resident
memory, and OpenBLAS reserves a stack per thread — across 64 threads that overruns any
sane ceiling before a single array is allocated, which is why a first attempt at a 4 GB
limit broke numpy outright.

`web_search` uses DuckDuckGo via `ddgs` with a Wikipedia API fallback. It is keyless, so
results are weaker than a production search backend — a real caveat when comparing
against published with-tools numbers.

## Files

`main.py` CLI · `dataset.py` subset selection and prompts · `runner.py` the two
generation modes · `tools.py` the python sandbox and search · `endpoint.py` async client
(vendored) · `resultdir.py` results-directory contract (vendored).
