---
name: judge-aalcr
description: Grade the answers in an AA-LCR benchmark results directory. Use when asked to judge, grade, or score an AA-LCR long-context run produced by eai-bench/aalcr. Grades one question per isolated subagent and writes verdicts into the results directory.
---

# Judging an AA-LCR run

You are grading the answers in one `aalcr/results/<timestamp>/` directory. The benchmark
client never grades: it only produces `responses.jsonl`. Your job is to turn those into
`grades/<safe_id>.json`, then `grades.jsonl` and `score.json`.

`$AALCR` below is `<repo>/aalcr/src/main.py` and `$DIR` is the results directory.

## The one rule that matters

**Every question is graded by its own subagent, and that subagent sees exactly one
question.** A grader that has seen another item's gold answer is contaminated. Do not
batch questions into one agent, and do not read `subset.json` or `responses.jsonl`
yourself to "check" a verdict.

## Procedure

1. Get the work list:

   ```
   python3 $AALCR pending --results-dir $DIR
   ```

   A JSON array of ids with a usable response and no grade yet; empty means done. The
   ids look like `<document_set_id>#<question_id>`, so quote them in the shell.

2. Spawn **one subagent per id**, in parallel, with the prompt below and `<ID>`
   replaced. Nothing else.

3. If a subagent fails, re-run `pending` and spawn fresh subagents for what is still
   listed rather than recording a verdict on its behalf.

4. Collect:

   ```
   python3 $AALCR collect --results-dir $DIR
   ```

   This writes `grades.jsonl` and `score.json`, including a per-document-category
   breakdown — the subset deliberately spans all seven categories. It warns on stderr if
   anything is ungraded; ungraded items count as incorrect.

## The grader prompt

> You are an equality checker for one question of the AA-LCR long-context benchmark.
> You grade EXACTLY ONE item and you have no knowledge of any other item. Do not look
> for others.
>
> **STEP 1 — fetch your item.** Run this exact command:
>
> ```
> python3 $AALCR show --results-dir $DIR '<ID>'
> ```
>
> It prints QUESTION, GOLD ANSWER and MODEL RESPONSE.
>
> **STEP 2 — decide correct / incorrect.** You are an equality checker, not a teacher.
> The only question is whether the model's FINAL committed answer means the same thing
> as the GOLD ANSWER.
>
> - Ignore formatting, markdown, bolding, currency symbols, thousands separators, units
>   spelled out vs abbreviated, and any surrounding explanation or derivation.
> - A number written differently but equal in value is CORRECT (e.g. gold "0.1" vs
>   answer "10 percentage points"; gold "50%" vs answer "50").
> - A sign or direction that contradicts the gold is INCORRECT.
> - If the gold is a set or ranking, the same members in the order the question asked
>   for is CORRECT even if the gold lists them in a different order.
> - If the model commits to an answer that is a superset or subset of the gold (extra or
>   missing items), that is INCORRECT.
> - If the model states the gold value along the way but then commits to a different
>   final conclusion, that is INCORRECT. Grade what it committed to at the end.
> - If the response is empty or never commits to an answer, that is INCORRECT.
> - Do not give credit for being "close" or for good reasoning. Match or no match.
>
> Read the WHOLE response before deciding — these responses often reason for a long time
> and reverse themselves near the end.
>
> **STEP 3 — record your verdict.** Set CORRECT to 1 or 0 and write one sentence for
> NOTE:
>
> ```
> printf '%s' "NOTE" | python3 $AALCR record --results-dir $DIR '<ID>' CORRECT
> ```
>
> It echoes the recorded JSON. If it does not, retry once. It refuses ids that are not
> in this run's subset — if that happens, report it rather than working around it.
>
> Report the id, your verdict, your one-sentence note, and whether the record command
> confirmed.
