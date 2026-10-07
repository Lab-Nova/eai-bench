"""The results-directory contract shared by every benchmark client.

A run lives entirely inside one directory:

    results/<UTC timestamp>/
      config.json      endpoint, model, sampling, mode, subset size, started_at
      subset.json      the exact items chosen -- the run's denominator
      responses.jsonl  one row per item, appended and flushed as it completes
      failures.jsonl   rows dropped by a resume, kept as an audit trail
      grades/<safe_id>.json   one verdict per item, written by the judge skill
      grades.jsonl     collected, in subset order
      score.json       the uniform summary synthesis.py reads
      run.log

Vendored byte-identically into `hle/src/` and `aalcr/src/`; see endpoint.py for why.
"""

import datetime
import json
import os
import random
import statistics

STAMP_FMT = "%Y-%m-%dT%H-%M-%SZ"


def now_stamp():
    """UTC, filesystem-safe, and lexically sortable -- so `max()` means `latest`."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(STAMP_FMT)


def safe_id(item_id):
    """A grade filename that survives ids containing slashes, spaces or quotes."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(item_id))


def is_done(row):
    """A row counts as done only if it actually produced an answer.

    The predecessor to this file resumed on id-presence alone, so a row written with
    `error != null` and an empty response counted as complete and was never retried.
    Rows had to be deleted by hand to get those items to run. Never again.
    """
    return not row.get("error") and bool((row.get("response") or "").strip())


def interaction_tokens(row):
    """How far one item grew the context: final context length minus prompt length.

    A client records it on the row as `interaction_tokens`. For a single completion it is
    the completion's token count; for a tool loop it is the last request's prompt plus
    completion minus the first request's prompt, so it counts the model's turns *and* the
    tool output fed back to it. Rows written before the field existed fall back to the
    single-turn reading of `usage`. None when nothing was measured.
    """
    t = row.get("interaction_tokens")
    if isinstance(t, (int, float)):
        return t
    usage = row.get("usage") or {}
    if "rounds" not in row and isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    return None


# The suite-wide per-request cap; a row that does not record its own `max_tokens` is
# read against this.
OUTPUT_CAP = 131072


def hit_output_limit(row):
    """Whether a row ran into a length limit: the token cap or the context window."""
    cap = row.get("max_tokens") or OUTPUT_CAP
    used = row.get("interaction_tokens") or 0
    return (row.get("finish_reason") == "length" or bool(row.get("context_full"))
            or used >= 0.99 * cap)


def no_answer_kind(row):
    """Why an error-free row has no final answer: 'output_limit', 'empty', or None.

    endpoint.finish_row stamps `no_answer` on new rows (and writes a NO ANSWER marker as
    the response). Rows written before that have an empty response instead and are read
    the same way here, so old and new runs report runaways alike.
    """
    if row.get("error"):
        return None
    if row.get("no_answer"):
        return row["no_answer"]
    if (row.get("response") or "").strip():
        return None
    return "output_limit" if hit_output_limit(row) else "empty"


def token_stats(values, n_boot=2000, seed=0):
    """n, median with a bootstrap SE, mean, p90 and max of token counts; None if empty.

    The SE is the stdev of the median over `n_boot` resamples (seeded, so a re-collect
    reproduces it).
    """
    v = sorted(values)
    if not v:
        return None
    rng = random.Random(seed)
    meds = [statistics.median(rng.choices(v, k=len(v))) for _ in range(n_boot)]
    return {"n": len(v), "median": statistics.median(v),
            "median_se": round(statistics.stdev(meds), 1) if len(v) > 1 else 0.0,
            "mean": round(statistics.mean(v), 1),
            "p90": v[min(len(v) - 1, int(0.9 * len(v)))], "max": v[-1]}


def context_lengths(usages):
    """(prompt, final_context, interaction) tokens from one item's per-request usages.

    `usages` are the usage dicts of the requests of one conversation, in order. The first
    request's prompt is the prompt; the last request's prompt + completion is the final
    context. (None, None, None) when no request reported usage.
    """
    usages = [u for u in usages if u and u.get("prompt_tokens") is not None]
    if not usages:
        return None, None, None
    prompt = usages[0]["prompt_tokens"]
    final = usages[-1]["prompt_tokens"] + (usages[-1].get("completion_tokens") or 0)
    return prompt, final, final - prompt


class ResultDir:
    """One run's directory. Create a new one with `start`, reopen one with `open`."""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    # -- lifecycle ---------------------------------------------------------------

    @classmethod
    def start(cls, base, suffix=None):
        """Mint a fresh timestamped directory under `base`."""
        name = now_stamp() + (f"-{suffix}" if suffix else "")
        rd = cls(os.path.join(base, name))
        os.makedirs(rd.path, exist_ok=True)
        os.makedirs(rd.grades_dir, exist_ok=True)
        return rd

    @classmethod
    def open(cls, path):
        rd = cls(path)
        if not os.path.isdir(rd.path):
            raise SystemExit(f"no such results directory: {rd.path}")
        os.makedirs(rd.grades_dir, exist_ok=True)
        return rd

    @classmethod
    def latest(cls, base, predicate=None):
        """The lexically greatest run under `base` satisfying `predicate`, or None."""
        if not os.path.isdir(base):
            return None
        for name in sorted(os.listdir(base), reverse=True):
            rd = cls(os.path.join(base, name))
            if not os.path.isdir(rd.path):
                continue
            if predicate is None or predicate(rd):
                return rd
        return None

    # -- paths -------------------------------------------------------------------

    def _p(self, *parts):
        return os.path.join(self.path, *parts)

    @property
    def config_path(self):
        return self._p("config.json")

    @property
    def subset_path(self):
        return self._p("subset.json")

    @property
    def responses_path(self):
        return self._p("responses.jsonl")

    @property
    def failures_path(self):
        return self._p("failures.jsonl")

    @property
    def grades_dir(self):
        return self._p("grades")

    @property
    def grades_path(self):
        return self._p("grades.jsonl")

    @property
    def score_path(self):
        return self._p("score.json")

    @property
    def log_path(self):
        return self._p("run.log")

    # -- json helpers ------------------------------------------------------------

    def _read_json(self, path, default=None):
        if not os.path.exists(path):
            return default
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    # -- config ------------------------------------------------------------------

    @property
    def config(self):
        return self._read_json(self.config_path, {}) or {}

    def write_config(self, cfg):
        self._write_json(self.config_path, cfg)

    # The fields that must not change silently mid-run: mixing two endpoints or two
    # subsets into one result set produces a number that describes nothing.
    IDENTITY = ("endpoint", "model", "mode", "n_subset")

    def reconcile(self, incoming, force=False):
        """Merge `incoming` into an existing config, refusing an identity change.

        Returns the config actually in force. With --force the run proceeds and the
        change is appended to `config_history`, so the artifact records the mixing
        instead of hiding it.
        """
        old = self.config
        if not old:
            self.write_config(incoming)
            return incoming
        drift = {k: (old.get(k), incoming.get(k)) for k in self.IDENTITY
                 if incoming.get(k) is not None and old.get(k) != incoming.get(k)}
        if drift and not force:
            lines = "\n".join(f"  {k}: {a!r} -> {b!r}" for k, (a, b) in drift.items())
            raise SystemExit(
                f"refusing to resume {self.path}: this run was started with different "
                f"settings.\n{lines}\n"
                f"Start a new run, or pass --force to append to it anyway (the change is "
                f"recorded in config.json).")
        if drift:
            history = old.setdefault("config_history", [])
            history.append({"at": now_stamp(),
                            "changed": {k: {"from": a, "to": b} for k, (a, b) in drift.items()}})
            old.update({k: incoming[k] for k in drift})
            self.write_config(old)
        return old

    # -- subset ------------------------------------------------------------------

    def write_subset(self, items, drop=("prompt",)):
        self._write_json(self.subset_path,
                         [{k: v for k, v in it.items() if k not in drop} for it in items])

    def read_subset(self):
        return self._read_json(self.subset_path, []) or []

    # -- responses ---------------------------------------------------------------

    def read_rows(self, path=None):
        """Every parseable row from a JSONL file, in file order."""
        path = path or self.responses_path
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write can leave one torn final line. Everything
                    # before it is intact, and the torn item simply reruns.
                    continue
        return rows

    def rows_by_id(self, path=None):
        """Last row wins, so a retried item reads as its most recent attempt."""
        return {str(r.get("id")): r for r in self.read_rows(path)}

    def prepare_resume(self):
        """Keep only completed rows in responses.jsonl; bank the rest in failures.jsonl.

        Returns (done_ids, n_dropped). Rewriting once, before any worker starts,
        preserves one-row-per-id for every reader downstream -- `pending`, `show`,
        `collect` and synthesis all get to stay trivial. A dropped row is by definition
        an empty answer plus an error string, so nothing of value leaves the directory.
        """
        rows = self.read_rows()
        if not rows:
            return set(), 0
        keep, drop, seen = [], [], set()
        for row in rows:
            rid = str(row.get("id"))
            if is_done(row) and rid not in seen:
                seen.add(rid)
                keep.append(row)
            else:
                drop.append(row)
        if not drop:
            return seen, 0
        tmp = self.responses_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, self.responses_path)
        with open(self.failures_path, "a", encoding="utf-8") as f:
            for row in drop:
                row["dropped_at"] = now_stamp()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return seen, len(drop)

    # -- grading -----------------------------------------------------------------

    def grade_path(self, item_id):
        return os.path.join(self.grades_dir, safe_id(item_id) + ".json")

    def read_grades(self):
        out = {}
        if not os.path.isdir(self.grades_dir):
            return out
        for name in sorted(os.listdir(self.grades_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.grades_dir, name), encoding="utf-8") as f:
                    g = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            out[str(g.get("id"))] = g
        return out

    def record_grade(self, item_id, correct, note=""):
        # A grader that mistakes an error string for an id would otherwise create a
        # verdict for an item that does not exist, quietly inflating the graded count.
        # The subset is the only legitimate id space, so check against it.
        known = {str(x["id"]) for x in self.read_subset()}
        if known and str(item_id) not in known:
            raise SystemExit(f"id is not in this run's subset, refusing to grade it: {item_id!r}")
        g = {"id": str(item_id), "correct": int(bool(correct)), "note": note,
             "graded_at": now_stamp()}
        self._write_json(self.grade_path(item_id), g)
        return g

    def pending_ids(self):
        """Ids with a usable response and no grade yet -- the judge skill's work list."""
        graded = self.read_grades()
        rows = self.rows_by_id()
        out = []
        for item in self.read_subset():
            rid = str(item["id"])
            if rid in graded:
                continue
            row = rows.get(rid)
            if row is not None and is_done(row):
                out.append(rid)
        return out

    def collect(self, component, extra=None, stratify=None):
        """Fold grades/ into grades.jsonl and score.json, in subset order.

        `stratify` (row -> key) is passed to runtime_stats for per-stratum interaction.
        """
        subset = self.read_subset()
        graded = self.read_grades()
        rows = self.rows_by_id()
        ordered = []
        for item in subset:
            rid = str(item["id"])
            g = graded.get(rid)
            if g is not None:
                ordered.append(g)
        with open(self.grades_path, "w", encoding="utf-8") as f:
            for g in ordered:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

        total = len(subset)
        correct = sum(1 for g in ordered if g.get("correct"))
        responded = sum(1 for item in subset
                        if is_done(rows.get(str(item["id"])) or {}))
        score = {
            "component": component,
            "correct": correct,
            "total": total,
            # The denominator is the subset, always. An item that never answered or was
            # never graded is not excused from it -- it is simply not correct.
            "accuracy": round(correct / total, 6) if total else 0.0,
            "graded": len(ordered),
            "responded": responded,
            "complete": len(ordered) == total,
            "results_dir": self.path,
            "scored_at": now_stamp(),
        }
        score.update(self.runtime_stats(stratify))
        cfg = self.config
        for k in ("endpoint", "model", "mode", "imported"):
            if k in cfg:
                score[k] = cfg[k]
        if extra:
            score.update(extra)
        self._write_json(self.score_path, score)
        return score

    # -- runtime -----------------------------------------------------------------

    def runtime_stats(self, stratify=None):
        """Latency, token totals and interaction statistics for the run.

        Interaction statistics (median with bootstrap SE, mean, p90, max) cover only the
        items that produced an answer. Runaways -- rows with no answer because they hit
        the output limit -- sit at the cap and would swamp the mean, so they are counted
        separately instead (`no_answer`, `output_limit_rate`). A pooled median cannot see
        a minority population (BFCL multi-turn, long qa3 lengths), so `stratify`
        (row -> key) adds the same numbers per stratum under `interaction_by`.
        """
        all_rows = self.read_rows()
        rows = [r for r in all_rows if is_done(r)]
        lat = sorted(r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float)))
        out = {}
        if lat:
            out["latency_s_median"] = round(statistics.median(lat), 2)
            out["latency_s_p90"] = round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 2)
            out["latency_s_max"] = round(lat[-1], 2)
        prompt_toks = completion_toks = 0
        for r in rows:
            usage = r.get("usage") or {}
            prompt_toks += usage.get("prompt_tokens") or 0
            completion_toks += (usage.get("completion_tokens")
                                or r.get("completion_tokens") or 0)
        if prompt_toks:
            out["prompt_tokens_total"] = prompt_toks
        if completion_toks:
            out["completion_tokens_total"] = completion_toks

        clean = [r for r in all_rows if not r.get("error")]
        answered = [r for r in clean if is_done(r) and not no_answer_kind(r)]
        out.update(interaction_summary(answered, clean))
        if stratify:
            keys = list(dict.fromkeys(stratify(r) for r in clean))
            out["interaction_by"] = {
                str(k): interaction_summary([r for r in answered if stratify(r) == k],
                                            [r for r in clean if stratify(r) == k])
                for k in keys}

        rounds = sorted(r["rounds"] for r in answered if isinstance(r.get("rounds"), int))
        if rounds:
            out["rounds_median"] = statistics.median(rounds)
            out["rounds_mean"] = round(statistics.mean(rounds), 1)
            out["rounds_p90"] = rounds[min(len(rounds) - 1, int(0.9 * len(rounds)))]
            out["rounds_max"] = rounds[-1]
            out["tool_calls_total"] = sum(r.get("tool_calls") or 0 for r in rows)
        return out


def interaction_summary(answered, clean):
    """Interaction stats over `answered` rows; no-answer counts and rate over `clean` rows.

    `clean` is every error-free row; `answered` the subset with a real answer.
    """
    out = {}
    st = token_stats([t for t in (interaction_tokens(r) for r in answered) if t is not None])
    if st:
        out["interaction_tokens_median"] = st["median"]
        out["interaction_tokens_median_se"] = st["median_se"]
        out["interaction_tokens_mean"] = st["mean"]
        out["interaction_tokens_p90"] = st["p90"]
        out["interaction_tokens_max"] = st["max"]
        out["interaction_tokens_n"] = st["n"]
    kinds = [no_answer_kind(r) for r in clean]
    out["no_answer"] = {"output_limit": kinds.count("output_limit"), "empty": kinds.count("empty")}
    out["output_limit_rate"] = round(kinds.count("output_limit") / len(clean), 4) if clean else 0.0
    return out
