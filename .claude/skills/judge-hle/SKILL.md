---
name: judge-hle
description: Grade the answers in an HLE benchmark results directory. Use when asked to judge, grade, or score an HLE run produced by eai-bench/hle, in either the with-tools or no-tools mode. Grades one question per isolated subagent and writes verdicts into the results directory.
---

# Judging an HLE run

You are grading the free-form answers in one `hle/results/<timestamp>/` directory. The
benchmark client never grades: it only produces `responses.jsonl`. Your job is to turn
those into `grades/<safe_id>.json`, then `grades.jsonl` and `score.json`.

`$HLE` below is `<repo>/hle/src/main.py` and `$DIR` is the results directory.

## The one rule that matters

**Every question is graded by its own subagent, and that subagent sees exactly one
question.** A grader that has seen another item's gold answer is contaminated. Do not
batch questions into one agent, do not read `subset.json` or `responses.jsonl` yourself
to "check" a verdict, and do not summarize several items together.

## Procedure

1. Get the work list:

   ```
   python3 $HLE pending --results-dir $DIR
   ```

   It prints a JSON array of ids that have a usable response and no grade yet. An empty
   array means the run is fully graded — go to step 4. Re-running the skill is safe:
   already-graded items never come back.

2. Grade the pending ids as **one large parallel workflow** — the Workflow tool, not
   agents dispatched by hand from the main loop. The script fans out **one subagent per
   id**, every pending id at once rather than in sequential batches, and each agent gets
   the prompt in *The grader prompt* below with `<ID>` replaced. Nothing else — no
   context about the run, the model, the other questions, or how the scoring is going.

   A workflow rather than loose agents because the fan-out is the point: 250 graders is
   a scale the main loop should not be babysitting one tool call at a time, and the
   script keeps the poll/dispatch/retry loop deterministic instead of model-driven.

   **Every grader runs on Sonnet** (`model: 'sonnet'` on the `agent()` call). This is not
   a cost note dressed up as a rule: the rubric is mechanical equality-checking against a
   gold string, a 250-item wave is 250 agents, and a grader that reasons its way to a
   more generous verdict is a worse grader. Pin the model explicitly rather than
   inheriting the caller's — the number must not move because the parent session
   happened to be running something else.

   To keep the swarm's launch cheap, write the grader prompt to one file and give each
   agent only its id plus the path — the agent reads the rubric itself, so isolation is
   preserved and the id is the only thing that varies.

   Grading does not have to wait for generation to finish: `pending` only ever lists ids
   that already have a usable response. So have the workflow **loop** — poll `pending`,
   fan out a wave, poll again — until the client process has exited and `pending` comes
   back empty. One workflow then drains the whole run alongside it, instead of one
   invocation per wave. Bound the retries (three attempts per id) so a grader that keeps
   failing to record cannot spin the loop forever.

3. If a subagent fails or reports that it could not record, re-run `pending` and spawn
   fresh subagents for whatever is still listed. Do not record a verdict on a grader's
   behalf from what it told you; the verdict has to come from the isolated agent that
   read the item.

4. Collect:

   ```
   python3 $HLE collect --results-dir $DIR
   ```

   This writes `grades.jsonl` and `score.json`. It warns on stderr if anything is still
   ungraded — ungraded items count as incorrect, so that warning means the number is a
   floor, not a result. Then optionally:

   ```
   python3 $HLE audit --results-dir $DIR
   ```

   which checks that no item is marked correct with an empty response, and — for a
   with-tools run — reports whether any tool trace touched a dataset-hosting domain.

## The grader prompt

> You are the equality checker for ONE question of Humanity's Last Exam (HLE).
> You grade EXACTLY ONE item and know nothing about any other item. Do not go looking
> for others.
>
> **STEP 1 — fetch your item.** Run this exact command:
>
> ```
> python3 $HLE show --results-dir $DIR '<ID>'
> ```
>
> It prints QUESTION, GOLD ANSWER and MODEL RESPONSE. The response follows the official
> HLE format: an "Explanation:" section, then an "Exact Answer:" (or "Answer:") line,
> then "Confidence:". If the run used tools, the model had a Python interpreter and web
> search available, so the explanation may cite computations or sources. That changes
> nothing about how you grade: you check the final answer against the gold answer and
> nothing else.
>
> **STEP 2 — extract the model's final answer** from the "Exact Answer:" / "Answer:"
> line. If that line is missing, use whatever the response commits to as its final
> answer.
>
> **STEP 3 — decide correct / incorrect.** You are an equality checker, NOT a grader of
> reasoning. The ONLY question is whether the extracted answer means the same thing as
> the GOLD ANSWER.
>
> - Ignore formatting entirely: LaTeX vs plain text, markdown, bolding, whitespace.
> - Mathematically equal forms are CORRECT: 1/2 = 0.5 = 50%; \frac{\pi}{2} = pi/2;
>   2\sqrt{3} = 2*sqrt(3); 1.0e-3 = 0.001. Equivalent algebraic rearrangements are
>   CORRECT.
> - A rounded value matching the gold's precision is CORRECT; a value that differs
>   beyond rounding is INCORRECT (e.g. gold 0.08 vs answer 1/21 = 0.0476 is INCORRECT).
> - Units: the same quantity in different units is CORRECT only if numerically
>   consistent.
> - For multiple choice, the chosen option is CORRECT if it is the gold option, whether
>   given as the letter, the option text, or both.
> - Chemical names / IUPAC / formulas, gene and species names, and other domain
>   notations count as equal if they denote the same entity.
> - Ignore the Confidence value entirely; it never affects correctness.
> - If the answer is a list or tuple, all elements must match (order matters only if the
>   question asked for an order).
> - Partially correct is INCORRECT. Being "close" or well-reasoned earns nothing.
> - An empty response, no committed answer, or a refusal is INCORRECT.
>
> Be strict but fair: the standard is "would a careful expert say these two answers are
> the same answer?" — not "did the model do good work?".
>
> **STEP 4 — record your verdict.** Replace CORRECT with 1 or 0, and NOTE with one
> sentence:
>
> ```
> printf '%s' "NOTE" | python3 $HLE record --results-dir $DIR '<ID>' CORRECT
> ```
>
> It echoes the recorded JSON. If it does not, retry once. It refuses ids that are not
> in this run's subset — if that happens, report it rather than working around it.
>
> Report the id, the answer you extracted, your verdict, your one-sentence note, and
> whether the record command confirmed.
