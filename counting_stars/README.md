# Counting-Stars

The correction test ("reasoning", Me-Rea.) of
[Counting-Stars](https://github.com/nick7nlp/Counting-Stars), in English and Chinese, at
15 lengths from 64k to 960k tokens. Each prompt hides 32 evenly spaced needles in a
haystack:

> The little penguin counted 16 ★, but found that a mistake had been made, so the
> counting was done again, and this time 15 ★ was counted correctly.

The question at the end asks for the list of *correct* counts as JSON. Recalling all 32 is
not enough. The model has to keep the corrected count and drop the wrong one, at every
depth of the context. On GLM-5.3 this separated quantizations that single-needle tests
rate identically.

The standard run is **2 languages × 15 lengths × 10 samples = 300 requests** (avg@10). Upstream
has one star set per needle count, so each (language, length) has exactly one prompt. A
sample is an independent request for that same prompt at the client's sampling
temperature. A single 32-needle prompt is noisy: one cell moved by up to 0.3 between
endpoints whose overall means agreed. That noise is why there are ten samples per cell
(five before 2026-10-05).

## Running

```bash
python3 counting_stars/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3
python3 counting_stars/src/main.py run --results-dir counting_stars/results/<ts>    # resume
python3 counting_stars/src/main.py run ... --langs EN --lengths 64k 512k --n-samples 1 # subset
python3 counting_stars/src/main.py generate                                        # prompts only
```

Grading is mechanical, so `run` ends by running `collect`, which prints the
length × language table. `show --results-dir DIR EN-960k-s0` prints one response and its
per-needle tally.

Settings:
- `--concurrency` defaults to 16.
- `--max-tokens` defaults to 131,072, the suite-wide per-request budget; `context` asks
  for the whole window minus the prompt, as in aalcr. The answer itself is short, but at
  960k the reasoning before it runs to tens of thousands of tokens. `collect` prints and
  `score.json` records the median `interaction_tokens` (final context − prompt).
  Past about 917k the cap is lowered to `--context-len` − prompt − 512 so the request
  fits the window (sglang refuses prompt + `max_tokens` over it); the cap actually sent
  is recorded per row as `max_tokens`.
- Install `tokenizers`, `datasets` and `openai`.

## Scoring

The scoring follows upstream's `viz.ipynb`. The predicted list is taken after the last
`little_penguin` / `小企鹅` key, or else the last `[...]` in the answer. It is cut to 32
entries and de-duplicated. Each needle then scores:

| Score | Condition |
|---|---|
| 1 | only the correct count is in the list |
| 0.5 | both the correct and the wrong count are |
| 0.25 | only the wrong count is |
| 0 | neither |

An item's score is the mean over its 32 needles. `accuracy` in `score.json` is the mean
item score. `accuracy_se` is the standard error over the 30 (language, length) cell
means: samples of one prompt are not independent, so the error is not taken over 300
requests. `score.json` also carries `by_lang`, `by_length` and needle totals. An item
with no answer scores 0 and stays in the denominator.

## Prompts

The prompts are byte-identical to the ones the reference numbers were measured on.
Two things differ from the 128k files shipped upstream, and both are needed to go past
128k:

- **Lengths are GLM-5.3 tokens** of the user message, cut with the model's own
  tokenizer, not words or characters. The tokenizer is fetched from `zai-org/GLM-5.3`
  and pinned by sha256.
- **English filler is PG19** (`emozilla/pg19-test`). Upstream's Paul Graham essays run
  out at about 140k tokens. Chinese filler is upstream's own (The Story of the Stone +
  Journey to the West), cleaned as in its `gen_test_data.ipynb`.

Needle i sits after filler token `(i+1)·n/32`, so the last needle is right before the
question. English cut points move to the next space.

Every input is pinned to a commit or revision and fetched into `counting_stars/data/`,
which is git-ignored. Built prompts are cached there too, about 60 MB for all 30.

## Reference numbers

GLM-5.3 BF16 weights / BF16 KV cache, 10 samples per cell (300 requests), temperature 0.6 /
top_p 0.95, `max_tokens` 131072, October 2026, mean ± SE over the 30 cells:
**0.887 ± 0.019** (EN 0.915, ZH 0.860).

| Length | 64k | 128k | 192k | 256k | 320k | 384k | 448k | 512k | 576k | 640k | 704k | 768k | 832k | 896k | 960k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EN+ZH | 0.989 | 0.955 | 0.969 | 0.992 | 0.941 | 0.886 | 0.961 | 0.944 | 0.863 | 0.884 | 0.900 | 0.758 | 0.809 | 0.722 | 0.739 |

Interaction tokens over the 293 answered requests: median 1,719 ± 60 (bootstrap SE),
mean 2,273, p90 3,047. 7 of 300 answers ran past the cap and scored 0.

## Files

`main.py` CLI · `dataset.py` inputs and prompt construction · `scoring.py` answer parsing
and needle scores · `endpoint.py`, `resultdir.py` vendored.
