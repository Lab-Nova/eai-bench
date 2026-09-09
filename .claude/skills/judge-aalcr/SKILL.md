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

2. Grade as **one large parallel workflow** (the Workflow tool), fanning out **one
   subagent per id** — every pending id at once, not in sequential batches — with the
   prompt below and `<ID>` replaced. Nothing else.

   **Every grader runs on Sonnet** (`model: 'sonnet'` on the `agent()` call), for the same
   reason as in `judge-hle`: the rubric is mechanical equality-checking, the wave is one
   agent per item, and the verdict must not depend on which model the parent session
   happened to be running. Pin it explicitly rather than inheriting.

   To keep the swarm's launch cheap, write the grader prompt to one file and give each
   agent only its id plus the path — the agent reads the rubric itself, so isolation is
   preserved and the id is the only thing that varies.

   The full 100 items fit in a single wave (16 graders run at a time, so it takes a few
   minutes) — the loop `judge-hle` needs is usually overkill here. The same trick applies
   if you are grading alongside a still-running client: `pending` only lists ids that
   already have a usable response, so the workflow can poll and fan out again rather than
   waiting for the run to end.

3. If a subagent fails, re-run `pending` and spawn fresh subagents for what is still
   listed rather than recording a verdict on its behalf.

4. Collect:

   ```
   python3 $AALCR collect --results-dir $DIR
   ```

   This writes `grades.jsonl` and `score.json`, including a per-document-category
   breakdown across the seven document categories. It warns on stderr if anything is
   ungraded; ungraded items count as incorrect.

## Which rubric

The grader prompt below carries the dataset's own **version 1.1 judge prompts verbatim**
(dataset card, "Scoring Approach", September 2026). Artificial Analysis runs them on
GPT-5.6 Luna at medium reasoning effort; here Sonnet stands in for that judge, and the
prompts are not to be paraphrased, softened or extended. Two consequences worth knowing
before you compare numbers:

- The v1.1 rubric is **format-strict**: where the question asks for a particular format
  (a percentage, a number of decimal places, a unit, a rounding, an ordering) the answer
  must meet it. 41 of the 100 questions ask for one. An older, more lenient in-house
  rubric ("ignore formatting…") gave a few points more; grades produced under it are not
  comparable with these.
- v1.1 also corrected 16 answer keys relative to v1.0.0, so scores against the two key
  versions are not comparable either.

## The grader prompt

> You are the equality checker for one question of the AA-LCR long-context benchmark,
> running the dataset's own version 1.1 judge prompts
> (https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR, "Scoring Approach"). You
> grade EXACTLY ONE item and have no knowledge of any other item. Your item id is given
> by the caller; write it in place of `<ID>` below (ids look like
> `<document_set_id>#<question_id>`, keep the single quotes).
>
> **STEP 1 — fetch your item.** Run this exact command:
>
> ```
> python3 $AALCR show --results-dir $DIR '<ID>'
> ```
>
> It prints QUESTION (= the question), GOLD ANSWER (= the OFFICIAL ANSWER) and MODEL
> RESPONSE (= the CANDIDATE ANSWER). Read the whole response before deciding.
>
> **STEP 2 — apply the official judge instructions, verbatim below.**
>
> System prompt:
>
> ```
> Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
> Note the following points when assessing correctness:
>
> - Numbers should still match when they are the same value written differently, e.g., a
>   percentage, a count of percentage points, and the equivalent decimal fraction are the same
>   value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
>   (thousand, million, bn) and different notations (thousands separators, currency symbols,
>   LaTeX markup, and numbers written as words).
> - Where the question asks for a particular format (e.g., a percentage, a number of decimal
>   places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
>   question asks for no particular format, accept any equivalent form.
> - In cases where the question asks for an ordered list, a title, honorific or article added
>   to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
>   if it is correct either with those additions or without them.
> - Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
>   reached while working, and alternatives it considers and sets aside, do not count. If it
>   offers several values without selecting one, it is incorrect even if one of them is right.
>   Hedging is fine as long as one clearly definitive answer is given.
> ```
>
> User prompt:
>
> ```
> Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
> For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.
>
> The question, for reference only: START QUESTION {question}
>
> END QUESTION
>
> The OFFICIAL ANSWER: {official_answer}
>
> END OFFICIAL ANSWER
>
> BEGIN CANDIDATE ANSWER TO ASSESS
>
> {candidate_answer}
>
> END CANDIDATE ANSWER TO ASSESS
>
> Reply as JSON, with a verdict of CORRECT or INCORRECT.
> ```
>
> An empty MODEL RESPONSE is INCORRECT. Nothing beyond these instructions applies: no
> credit for being close or for good reasoning, and no penalty for extra explanation around
> a committed answer that meets the OFFICIAL ANSWER and any format the question asks for.
>
> **STEP 3 — record your verdict.** CORRECT is 1, INCORRECT is 0; NOTE is one sentence
> saying which rule decided it:
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
