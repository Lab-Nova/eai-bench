# AA-LCR-25

25 questions from [AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR),
a long-context recall benchmark: each prompt carries a full set of source documents and
asks a question that can only be answered by reading them.

Artificial Analysis does not publish which 25 of the 100 it uses, so this takes its own,
without an RNG: sort by `(document_category, document_set_id, question_id)` and stride
evenly. That spreads the sample across all seven document categories instead of
clustering on the 63-question Company set. `collect` reports the per-category breakdown.

## Running

```bash
python3 aalcr/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3
python3 aalcr/src/main.py run --results-dir aalcr/results/<ts>       # resume
```

The dataset is public (Apache-2.0) and fetched into `aalcr/data/` on first run — the CSV
plus a zip of extracted document text, about 17 MB, git-ignored. Point `--data-dir` at an
existing copy to skip the download.

`--concurrency` defaults to **8**, deliberately low: each prompt is hundreds of thousands
of tokens, so a high concurrency here is a memory problem, not a throughput win.
`--max-tokens` defaults to 24,576, which is generous — the documents are enormous but the
answer is a figure or a sentence.

Then grade with the **judge-aalcr** skill, which ends by running `collect`.

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
