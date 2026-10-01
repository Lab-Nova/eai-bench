"""Counting-Stars scoring, as upstream's viz.ipynb (reduce_duplicate + get_reasoning_score).

The predicted list is cut to 32 entries and de-duplicated. Then, per needle:

    1     only the correct count is in the list
    0.5   both the correct and the wrong count are
    0.25  only the wrong count is
    0     neither

A prompt's score is the mean over its 32 needles.
"""

import re

M = 32
ANSWER_KEY = {"EN": "little_penguin", "ZH": "小企鹅"}


def parse_counts(text, lang):
    """Integers of the list after the last answer key, else of the last [...] list.

    None if the text holds no list at all.
    """
    pos = text.rfind(ANSWER_KEY[lang])
    m = re.search(r"\[([^\[\]]*)\]", text[pos:]) if pos != -1 else None
    if m is not None:
        body = m.group(1)
    else:
        lists = re.findall(r"\[([^\[\]]*)\]", text)
        if not lists:
            return None
        body = lists[-1]
    return [int(x) for x in re.findall(r"-?\d+", body)]


def needle_scores(pred, a_stars, r_stars):
    pred = set((pred or [])[:M])
    scores = []
    for a, r in zip(a_stars, r_stars):
        if a in pred and r in pred:
            scores.append(0.5)
        elif a in pred:
            scores.append(1.0)
        elif r in pred:
            scores.append(0.25)
        else:
            scores.append(0.0)
    return scores


def grade(response, item):
    """Score one response against its item; an empty or list-less response scores 0."""
    pred = parse_counts(response or "", item["lang"])
    s = needle_scores(pred, item["reference"], item["wrong"])
    return {"score": sum(s) / M,
            "n_correct": s.count(1.0), "n_both": s.count(0.5),
            "n_wrong": s.count(0.25), "n_missing": s.count(0.0),
            "n_pred": -1 if pred is None else len(pred)}
