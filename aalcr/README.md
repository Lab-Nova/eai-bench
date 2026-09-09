# AA-LCR

All 100 questions of [AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR),
a long-context reasoning benchmark: each prompt carries a full set of source documents
(240k-550k characters) and asks a question that can only be answered by reading them.

The client fetches the dataset's `main` revision, which is **version 1.1** (September
2026): 16 answer keys were corrected there and a judge system prompt was added, and the
card says v1.1 scores are not comparable with v1.0.0 scores. `--n-subset N` runs a
deterministic subset instead — sort by `(document_category, document_set_id, question_id)`
and stride evenly, which spreads a sample across all seven document categories instead of
clustering on the 63-question Company set. `collect` reports the per-category breakdown.

## Running

```bash
python3 aalcr/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3
python3 aalcr/src/main.py run --results-dir aalcr/results/<ts>       # resume
```

The dataset is public (Apache-2.0) and fetched into `aalcr/data/` on first run — the CSV
plus a zip of extracted document text, about 17 MB, git-ignored. Point `--data-dir` at an
existing copy to skip the download.

`--concurrency` defaults to **0**, which means the whole subset at once — all 100 items
with the default `--n-subset`. Each prompt is hundreds of thousands of tokens, so the server's
prefill queue, not this flag, is what actually paces the run; set a positive value to
bound it. The resolved number (not the sentinel) is what lands in `config.json`.
`--max-tokens` defaults to `context`: each request asks for the whole window, `--context-len`
(1,048,576) minus the prompt's tokens as the server itself counts them (it states the count when
refusing a deliberately over-long probe, which costs nothing). The answer is a figure or a
sentence, but the reasoning before it is not small — under the old fixed 24,576 cap 3% of
answers were all reasoning and no answer. Pass an integer to cap it anyway.

Then grade with the **judge-aalcr** skill, which ends by running `collect`. The skill
carries the dataset's own v1.1 judge prompts verbatim (Artificial Analysis runs them on
GPT-5.6 Luna at medium effort; the skill runs them on Sonnet, one isolated grader per item).

## Two things that will bite

**Filenames in the zip are mojibake.** Some names store U+2019 mangled, so matching the
CSV's filenames against the extracted tree by exact string fails. `_key()` compares only
the ASCII alphanumerics, which survives every such mangling, and raises if two files in
one directory collapse to the same key.

**Prompts are not stored.** `subset.json` keeps the metadata and gold answers but not the
multi-megabyte document text, so a resume rebuilds the prompts from the local corpus. The
corpus must therefore still be present (or fetchable) to resume a run.

The prompt template — `BEGIN INPUT DOCUMENTS` / per-document `BEGIN DOCUMENT n:` markers
/ `START QUESTION` — follows the dataset card verbatim. Do not reformat it casually; the
document framing is part of what the benchmark measures.

## Files

`main.py` CLI · `dataset.py` fetching, subset selection, document assembly ·
`endpoint.py` async client (vendored) · `resultdir.py` results-directory contract
(vendored).
