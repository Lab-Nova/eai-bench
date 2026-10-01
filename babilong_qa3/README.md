# BABILong qa3

Task qa3 ("three supporting facts") of [BABILong](https://github.com/booydar/babilong),
with 100 samples at each of four haystack lengths (0k, 128k, 256k, 384k), 400 requests
in all. bAbI facts are scattered through PG19 text, and the question asks where an item
was *before* a given room:

> Mary grabbed the apple. … Mary went back to the bathroom. … Mary travelled to the
> kitchen. … Where was the apple before the kitchen?

To answer, the model has to find and chain three facts: who took the item, where they
went, and where they went next. On GLM-5.3 this was the BABILong task that separated
quantizations best. At 256k, FP4 weights with a poor recipe and a plain e4m3 KV cache
each lost about 15 points against BF16. qa1 moved 5 to 8 points, and qa5 and qa9 stayed
within noise. 0k is the no-haystack control.

## Running

```bash
python3 babilong_qa3/src/main.py run --endpoint http://127.0.0.1:30000/v1 --model GLM-5.3
python3 babilong_qa3/src/main.py run --results-dir babilong_qa3/results/<ts>     # resume
python3 babilong_qa3/src/main.py run ... --lengths 256k --n-per-length 20        # subset
python3 babilong_qa3/src/main.py generate                                        # data only
```

Grading is BABILong's own string match, so `run` ends by running `collect`, which prints
accuracy per length. `show --results-dir DIR qa3-256k-7` prints one item.

Settings:
- `--concurrency` defaults to 50.
- `--max-tokens` defaults to 131,072, the suite-wide per-request budget, which is also
  the cap the reference numbers below were measured under. qa3 at 256k is reasoned over
  for about 40k tokens on average, but the tail runs past 131k: 3 to 8 of 100 answers per
  endpoint were truncated to nothing. `score.json` records `truncated`, the number of
  answers that hit `max_tokens`, and the median `interaction_tokens` (final context −
  prompt). `--max-tokens context` asks for the whole window minus the prompt instead.
- Install `datasets`, `tokenizers`, `nltk`, `pandas`, `numpy` and `openai`. The last
  four are only needed for generation.

## Data

- **0k, 128k, 256k** are the official splits, fetched from HF `RMT-team/babilong` at a
  pinned revision. The first 100 samples of each are used.
- **384k is not published**, so the client generates it on first use with BABILong's own
  recipe (`create_tasks.py`). It uses seed 42 + 3·1000 + 384, qa3 facts from bAbI en-10k
  train, PG19 test noise, and `384000 − 300` GPT-2 tokens of message. The generator is
  vendored in `babi_noise.py` and is unchanged apart from dropping the torch base class.
  It is deterministic, so every machine gets the same 100 samples. These are the samples
  the reference numbers were measured on. Generation takes a few minutes.
- Lengths are GPT-2 tokens of haystack, as in BABILong. On GLM-5.3, 256k is about 250k
  prompt tokens and 384k about 375k.

Everything lands in `babilong_qa3/data/`, which is git-ignored.

## Prompt and scoring

The prompt is BABILong's default for qa3, verbatim from `babilong/prompts.py`: the
instruction, two in-context examples, and the post-prompt "Always return your answer in
the following format: Before the $location_1$ the $item$ was in the $location_2$."
Then come `<context>…</context>` and the question. The system prompt is "You are a
helpful AI assistant.".

Grading is `compare_answers` from `babilong/metrics.py`. Only the first sentence of the
answer counts. It is correct when the target room is the only qa3 room it mentions,
after dropping rooms the question itself names. Naming two rooms is wrong.

## Reference numbers

GLM-5.3, 100 samples, 256k, `max_tokens` 131072:

| Weights / KV cache | qa3 256k |
|---|---|
| BF16 / BF16 | 51% |
| FP8 / BF16 | 46% |
| MXFP4 (MR-GPTQ v1) / BF16 | 37% |
| MXFP4 (MR-GPTQ v2) / BF16 | 51% |
| MXFP4 (MR-GPTQ v2) / FP8 (FlashMLA, per-128 scaled) | 47% |
| MXFP4 (MR-GPTQ v2) / FP8 (plain e4m3, trtllm) | 36% |

Paired differences on the same 100 items have an SE of about 6 points, so a gap under
about 12 points is within noise.

## Files

`main.py` CLI · `dataset.py` splits, generation, prompt · `babi_noise.py` vendored
BABILong generator · `scoring.py` answer matching · `endpoint.py`, `resultdir.py`
vendored.
