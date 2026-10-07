#!/usr/bin/env python3
"""Checks for the parts of the clients that would otherwise need a live endpoint.

Everything here runs offline and in a temporary directory, except the last check, which
reads an imported results directory if one is present. Run it after any change to
resultdir.py -- the resume rule it verifies is the bug that cost the original run two
hand-repairs of a 21 MB JSONL file.

    python3 tools/selftest.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hle", "src"))

from resultdir import ResultDir, is_done  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name, cond, detail=""):
    print(f"{PASS if cond else FAIL}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_resume(tmp):
    """An id is done only with no error AND a non-empty response."""
    print("\nresume: only a real answer counts as done")
    rd = ResultDir.start(tmp)
    rd.write_subset([{"id": str(i), "question": "q", "gold": "g"} for i in range(5)],
                    drop=())
    write_rows(rd.responses_path, [
        {"id": "0", "response": "an answer", "error": None},
        {"id": "1", "response": "", "error": "APIConnectionError: connection refused"},
        {"id": "2", "response": "", "error": None},          # empty, no error
        {"id": "3", "response": "another", "error": None},
        {"id": "4", "response": None, "error": "TimeoutError"},
    ])
    check("is_done accepts a real answer", is_done({"id": "0", "response": "x", "error": None}))
    check("is_done rejects an errored row",
          not is_done({"id": "1", "response": "", "error": "APIConnectionError"}))
    check("is_done rejects an empty response",
          not is_done({"id": "2", "response": "   ", "error": None}))

    done, dropped = rd.prepare_resume()
    check("resume keeps only the answered ids", done == {"0", "3"}, f"got {sorted(done)}")
    check("resume drops the other three", dropped == 3, f"got {dropped}")
    kept = [r["id"] for r in rd.read_rows()]
    check("responses.jsonl now holds just the good rows", kept == ["0", "3"], f"got {kept}")
    banked = [r["id"] for r in rd.read_rows(rd.failures_path)]
    check("dropped rows are banked in failures.jsonl",
          sorted(banked) == ["1", "2", "4"], f"got {sorted(banked)}")

    # A second resume with nothing new to drop must not disturb the file.
    done2, dropped2 = rd.prepare_resume()
    check("a second resume is a no-op", (done2, dropped2) == ({"0", "3"}, 0))

    # A torn final line (killed mid-write) costs only that item.
    with open(rd.responses_path, "a", encoding="utf-8") as f:
        f.write('{"id": "9", "resp')
    check("a torn trailing line is skipped, not fatal",
          [r["id"] for r in rd.read_rows()] == ["0", "3"])
    return rd


def test_config(tmp):
    """A results dir refuses to mix two endpoints without --force."""
    print("\nconfig: an identity change aborts a resume")
    rd = ResultDir.start(tmp, suffix="cfg")
    base = {"endpoint": "http://a:30000/v1", "model": "GLM-5.3", "mode": "tools",
            "n_subset": 250}
    rd.reconcile(base)
    check("an unchanged config reconciles",
          rd.reconcile(dict(base))["endpoint"] == "http://a:30000/v1")

    changed = {**base, "endpoint": "http://b:30000/v1"}
    try:
        rd.reconcile(changed)
        check("a changed endpoint aborts", False, "no SystemExit raised")
    except SystemExit as e:
        check("a changed endpoint aborts", "refusing to resume" in str(e))

    try:
        rd.reconcile({**base, "model": "other"})
        check("a changed model aborts", False, "no SystemExit raised")
    except SystemExit:
        check("a changed model aborts", True)

    cfg = rd.reconcile(changed, force=True)
    check("--force proceeds", cfg["endpoint"] == "http://b:30000/v1")
    check("--force records the change in config_history",
          len(cfg.get("config_history", [])) == 1
          and cfg["config_history"][0]["changed"]["endpoint"]["to"] == "http://b:30000/v1")


def test_grading(tmp):
    """pending -> record -> collect, plus the guard on unknown ids."""
    print("\ngrading: pending / record / collect round-trip")
    rd = ResultDir.start(tmp, suffix="grade")
    rd.write_subset([{"id": str(i), "question": "q", "gold": "g"} for i in range(4)],
                    drop=())
    write_rows(rd.responses_path, [
        {"id": "0", "response": "a", "error": None},
        {"id": "1", "response": "b", "error": None},
        {"id": "2", "response": "", "error": "boom"},   # no answer -> not gradeable
        {"id": "3", "response": "d", "error": None},
    ])
    check("pending lists only answered, ungraded ids",
          rd.pending_ids() == ["0", "1", "3"], f"got {rd.pending_ids()}")
    rd.record_grade("0", 1, "right")
    check("a graded id leaves the pending list", rd.pending_ids() == ["1", "3"])

    try:
        rd.record_grade("docker: command not found", 1, "junk")
        check("record refuses an id outside the subset", False, "no SystemExit raised")
    except SystemExit as e:
        check("record refuses an id outside the subset", "not in this run's subset" in str(e))

    rd.record_grade("1", 0, "wrong")
    rd.record_grade("3", 1, "right")
    score = rd.collect("test")
    check("correct counts only the 1s", score["correct"] == 2, f"got {score['correct']}")
    check("the denominator is the subset, not the graded count",
          score["total"] == 4, f"got {score['total']}")
    check("an ungradeable item leaves the run incomplete", score["complete"] is False)
    check("accuracy divides by the subset", abs(score["accuracy"] - 0.5) < 1e-9)
    check("grades.jsonl follows subset order",
          [g["id"] for g in rd.read_rows(rd.grades_path)] == ["0", "1", "3"])


def test_imported_roundtrip():
    """Delete one verdict from a real imported run and put it back."""
    base = os.path.join(REPO, "aalcr", "results")
    if not os.path.isdir(base):
        print("\nimported run: none present, skipping")
        return
    dirs = [d for d in sorted(os.listdir(base))
            if os.path.exists(os.path.join(base, d, "score.json"))]
    if not dirs:
        print("\nimported run: none scored, skipping")
        return
    path = os.path.join(base, dirs[-1])
    print(f"\nimported run: regrade one item of {dirs[-1]}")
    main = os.path.join(REPO, "aalcr", "src", "main.py")

    def run(*args, stdin=None):
        return subprocess.run([sys.executable, main, *args], capture_output=True,
                              text=True, input=stdin)

    with open(os.path.join(path, "score.json"), encoding="utf-8") as f:
        before = json.load(f)
    check("the imported run starts fully graded",
          json.loads(run("pending", "--results-dir", path).stdout) == [])

    rd = ResultDir(path)
    victim = rd.read_subset()[0]["id"]
    gpath = rd.grade_path(victim)
    saved = open(gpath, encoding="utf-8").read()
    os.remove(gpath)
    try:
        pend = json.loads(run("pending", "--results-dir", path).stdout)
        check("pending returns exactly the deleted id", pend == [victim], f"got {pend}")
        old = json.loads(saved)
        r = run("record", "--results-dir", path, victim, str(int(bool(old["correct"]))),
                stdin=old.get("note", ""))
        check("record writes the verdict back", r.returncode == 0, r.stderr[-200:])
        run("collect", "--results-dir", path)
        with open(os.path.join(path, "score.json"), encoding="utf-8") as f:
            after = json.load(f)
        check("the score is unchanged after the round-trip",
              (after["correct"], after["total"]) == (before["correct"], before["total"]),
              f"{after['correct']}/{after['total']} vs {before['correct']}/{before['total']}")
    finally:
        with open(gpath, "w", encoding="utf-8") as f:
            f.write(saved)
        run("collect", "--results-dir", path)


def _collect(client, path):
    main = os.path.join(REPO, client, "src", "main.py")
    r = subprocess.run([sys.executable, main, "collect", "--results-dir", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-300:]
    with open(os.path.join(path, "score.json"), encoding="utf-8") as f:
        return json.load(f), ""


def test_counting_stars(tmp):
    """Mechanical scoring: 1 / 0.5 / 0.25 per needle, unanswered items score 0."""
    print("\ncounting_stars: needle scoring and collect")
    rd = ResultDir.start(tmp, suffix="cs")
    ref, wrong = list(range(10, 42)), list(range(110, 142))
    item = {"lang": "EN", "length": "64k", "prompt_tokens": 64000,
            "reference": ref, "wrong": wrong}
    rd.write_subset([{**item, "id": f"EN-64k-s{k}", "sample": k} for k in range(4)], drop=())
    write_rows(rd.responses_path, [
        # all correct, wrapped in prose
        {"id": "EN-64k-s0", "error": None,
         "response": 'Here: {"little_penguin": ' + json.dumps(ref) + "}"},
        # first 32 entries: ref[:16] + wrong[:16] -> 16 needles at 0.5, 16 at 0
        {"id": "EN-64k-s1", "error": None,
         "response": json.dumps({"little_penguin": ref[:16] + wrong})},
        # no list at all
        {"id": "EN-64k-s2", "error": None, "response": "I could not find any stars."},
        # never answered
        {"id": "EN-64k-s3", "error": "TimeoutError", "response": ""},
    ])
    score, err = _collect("counting_stars", rd.path)
    check("counting_stars collect runs", score is not None, err)
    if score is None:
        return
    g = {r["id"]: r["score"] for r in rd.read_rows(rd.grades_path)}
    check("a fully correct list scores 1", g.get("EN-64k-s0") == 1.0, f"got {g}")
    # the list holds 48 numbers but only the first 32 count: ref[:16] + wrong[:16]
    check("both counts score 0.5; entries past the 32nd are ignored",
          abs(g.get("EN-64k-s1", -1) - (16 * 0.5 + 0) / 32) < 1e-9, f"got {g.get('EN-64k-s1')}")
    check("no list scores 0", g.get("EN-64k-s2") == 0.0)
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "cs_scoring", os.path.join(REPO, "counting_stars", "src", "scoring.py"))
    sc = module_from_spec(spec)
    spec.loader.exec_module(sc)
    only_wrong = sc.grade(json.dumps({"little_penguin": wrong}), item)
    check("only the wrong counts score 0.25 per needle", only_wrong["score"] == 0.25,
          f"got {only_wrong}")
    check("an unanswered item scores 0 and stays in the denominator",
          g.get("EN-64k-s3") == 0.0 and score["total"] == 4)
    check("the run is incomplete while an item has no answer", score["complete"] is False)
    check("accuracy is the mean item score", abs(score["accuracy"] - 1.25 / 4) < 1e-6,
          f"got {score['accuracy']}")


def test_babilong_qa3(tmp):
    """BABILong's match: the target must be the only new label in the first sentence."""
    print("\nbabilong_qa3: question rewrite, answer matching and collect")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qa3_dataset", os.path.join(REPO, "babilong_qa3", "src", "dataset.py"))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    q = ds.clarify("Where was the apple before the kitchen? ")
    check("the upstream question is rewritten to name the last entry",
          q == "Where was the apple just before it was last carried into the kitchen?", f"got {q!r}")
    try:
        ds.clarify("Where is the apple?")
        check("an unexpected question shape is refused", False)
    except ValueError:
        check("an unexpected question shape is refused", True)
    rd = ResultDir.start(tmp, suffix="qa3")
    rd.write_subset([{"id": f"qa3-0k-{i}", "length": "0k", "idx": i, "question": q,
                      "target": "bathroom"} for i in range(5)], drop=())
    write_rows(rd.responses_path, [
        {"id": "qa3-0k-0", "error": None,
         "response": "Before the kitchen the apple was in the bathroom."},
        {"id": "qa3-0k-1", "error": None,                       # hedges across two rooms
         "response": "Before the kitchen the apple was in the bathroom or the garden."},
        {"id": "qa3-0k-2", "error": None,                       # right room, second sentence
         "response": "It is unclear. Before the kitchen the apple was in the bathroom."},
        {"id": "qa3-0k-3", "error": None,
         "response": "Before the kitchen the apple was in the office."},
        {"id": "qa3-0k-4", "error": "APIConnectionError", "response": ""},
    ])
    score, err = _collect("babilong_qa3", rd.path)
    check("babilong_qa3 collect runs", score is not None, err)
    if score is None:
        return
    g = {r["id"]: r["correct"] for r in rd.read_rows(rd.grades_path)}
    check("the formatted answer is correct (the question's room is ignored)",
          g.get("qa3-0k-0") == 1, f"got {g}")
    check("naming two rooms is wrong", g.get("qa3-0k-1") == 0)
    check("only the first sentence counts", g.get("qa3-0k-2") == 0)
    check("the wrong room is wrong", g.get("qa3-0k-3") == 0)
    check("an unanswered item is ungraded and counts against the subset",
          "qa3-0k-4" not in g and score["correct"] == 1 and score["total"] == 5
          and score["complete"] is False)
    check("by_length carries the per-length tally",
          score.get("by_length") == {"0k": {"correct": 1, "total": 5}}, f"got {score.get('by_length')}")


def test_interaction(tmp):
    """max_tokens on every request, and interaction = final context - prompt."""
    print("\ninteraction tokens: definition, HLE tool loop, BFCL rows, runtime stats")
    import asyncio
    from types import SimpleNamespace as NS
    from resultdir import context_lengths, interaction_tokens
    import runner

    check("context_lengths: first prompt to last prompt + completion",
          context_lengths([{"prompt_tokens": 100, "completion_tokens": 50},
                           {"prompt_tokens": 300, "completion_tokens": 20}]) == (100, 320, 220))
    check("context_lengths of nothing is None", context_lengths([None]) == (None, None, None))
    check("a single-turn row falls back to its completion tokens",
          interaction_tokens({"usage": {"completion_tokens": 7}}) == 7)
    check("a multi-turn row without the field is not guessed",
          interaction_tokens({"rounds": 3, "usage": {"completion_tokens": 7}}) is None)

    class Fake:
        """Round 1 calls a tool, round 2 answers."""
        def __init__(self):
            self.kwargs = []

        async def create(self, messages, **kw):
            self.kwargs.append(kw)
            n = len(self.kwargs)
            usage = NS(completion_tokens=50 if n == 1 else 20,
                       model_dump=lambda: {"prompt_tokens": 100 if n == 1 else 300,
                                           "completion_tokens": 50 if n == 1 else 20})
            calls = [NS(id="c1", function=NS(name="nope", arguments="{}"))] if n == 1 else None
            msg = NS(content=None if n == 1 else "42", tool_calls=calls)
            return NS(choices=[NS(message=msg)], usage=usage)

    fake = Fake()
    row = asyncio.run(runner.run_with_tools(fake, {"id": "x", "prompt": "q"}))
    check("hle: every tool-loop request carries max_tokens 131072",
          [k.get("max_tokens") for k in fake.kwargs] == [131072, 131072], f"got {fake.kwargs}")
    check("hle: interaction spans the item's own conversation",
          (row.get("prompt_tokens"), row.get("final_context_tokens"),
           row.get("interaction_tokens")) == (100, 320, 220), f"got {row}")

    class Greedy:
        """Asks for two tool calls every round until told to stop, then answers."""
        def __init__(self):
            self.convos = []

        async def create(self, messages, **kw):
            self.convos.append([dict(m) for m in messages])
            n = len(self.convos)
            usage = NS(completion_tokens=1, model_dump=lambda: {"prompt_tokens": 10 * n,
                                                                "completion_tokens": 1})
            if kw.get("tool_choice") == "none":
                msg = NS(content="7", tool_calls=None)
            else:
                msg = NS(content=None, tool_calls=[
                    NS(id=f"r{n}a", function=NS(name="nope", arguments="{}")),
                    NS(id=f"r{n}b", function=NS(name="nope", arguments="{}"))])
            return NS(choices=[NS(message=msg)], usage=usage)

    greedy = Greedy()
    row = asyncio.run(runner.run_with_tools(greedy, {"id": "y", "prompt": "q"}, max_tool_calls=3))
    tool_msgs = [m for m in greedy.convos[-1] if m["role"] == "tool"]
    check("hle: the default tool-call budget is 512", runner.DEFAULT_MAX_TOOL_CALLS == 512)
    check("hle: only the budgeted tool calls execute",
          row.get("tool_calls") == 3 and len(row.get("tool_trace") or []) == 3, f"got {row}")
    check("hle: every tool result reports the budget left",
          [m["content"].split("[tool budget: ")[-1].split("]")[0] for m in tool_msgs]
          == ["1 of 3 tool calls used, 2 remaining", "2 of 3 tool calls used, 1 remaining",
              "3 of 3 tool calls used, 0 remaining", "3 of 3 tool calls used, 0 remaining"],
          f"got {[m['content'] for m in tool_msgs]}")
    check("hle: a call past the budget is answered, not run",
          tool_msgs[-1]["content"].startswith("Not executed.") and "exhausted" in tool_msgs[-1]["content"])
    check("hle: an exhausted budget goes to the forced final with an answer",
          row.get("tool_budget_hit") is True and row.get("forced_final") is True
          and row.get("response") == "7", f"got {row}")
    row = asyncio.run(runner.run_with_tools(Fake(), {"id": "z", "prompt": "q"}, max_tool_calls=0))
    check("hle: max_tool_calls=0 leaves the loop uncapped",
          row.get("tool_budget_hit") is False and row.get("response") == "42", f"got {row}")

    class Thinker:
        """Round 1 reasons and calls a tool, round 2 answers; returns reasoning in `field`."""
        def __init__(self, field):
            self.field, self.convos = field, []

        async def create(self, messages, **kw):
            self.convos.append([dict(m) for m in messages])
            n = len(self.convos)
            usage = NS(completion_tokens=5, model_dump=lambda: {"prompt_tokens": 10 * n,
                                                               "completion_tokens": 5})
            calls = [NS(id="t1", function=NS(name="nope", arguments="{}"))] if n == 1 else None
            msg = NS(content=None if n == 1 else "42", tool_calls=calls, **{self.field: f"plan {n}"})
            return NS(choices=[NS(message=msg)], usage=usage)

    def sent_back(field, **kw):
        thinker = Thinker(field)
        asyncio.run(runner.run_with_tools(thinker, {"id": "w", "prompt": "q"}, **kw))
        return next(m for m in thinker.convos[-1] if m["role"] == "assistant")

    check("hle: an assistant turn goes back with its reasoning_content",
          sent_back("reasoning_content").get("reasoning_content") == "plan 1")
    check("hle: reasoning goes back under the field the server returned it in",
          sent_back("reasoning").get("reasoning") == "plan 1"
          and "reasoning_content" not in sent_back("reasoning"))

    sys.path.insert(0, os.path.join(REPO, "bfcl", "src"))
    try:
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("bfcl_main", os.path.join(REPO, "bfcl", "src", "main.py"))
        bm = module_from_spec(spec)
        spec.loader.exec_module(bm)
        check("bfcl: single-turn interaction is the output",
              bm.row_interaction_tokens({"input_token_count": 900, "output_token_count": 40}) == 40)
        check("bfcl: multi-turn interaction is last in + last out - first in",
              bm.row_interaction_tokens({"input_token_count": [[900, 1000], [1500]],
                                         "output_token_count": [[30, 40], [60]]}) == 660)
    finally:
        sys.path.remove(os.path.join(REPO, "bfcl", "src"))

    # A retried multi-turn test must start from its scenario, not from the state its
    # failed attempt left in bfcl-eval's cached environment. Needs the bfcl venv.
    probe = (
        "import sys; sys.path.insert(0, 'bfcl/src'); import shim\n"
        "from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import "
        "execute_multi_turn_func_call as ex\n"
        "cfg = {'GorillaFileSystem': {'root': {'w': {'type': 'directory', 'contents': {}}}}}\n"
        "ex(['mkdir(dir_name=\"stale\")'], cfg, ['GorillaFileSystem'], 'm', 't')\n"
        "shim.reset_multi_turn_state()\n"
        "print(ex(['ls()'], cfg, ['GorillaFileSystem'], 'm', 't')[0][0])\n")
    venv_py = os.path.join(REPO, "bfcl", ".venv", "bin", "python")
    p = (subprocess.run([venv_py, "-c", probe], cwd=REPO, capture_output=True, text=True)
         if os.path.exists(venv_py) else None)
    if p is None or "No module named" in p.stderr or "cannot execute" in p.stderr:
        print("  skip  bfcl: multi-turn reset (bfcl/.venv not usable on this host)")
    else:
        check("bfcl: a reset multi-turn environment starts from its scenario",
              p.returncode == 0 and '"current_directory_content": []' in p.stdout,
              (p.stdout + p.stderr)[-300:])

    rd = ResultDir.start(tmp, suffix="it")
    write_rows(rd.responses_path, [
        {"id": "a", "error": None, "response": "x", "usage": {"completion_tokens": 10}},
        {"id": "b", "error": None, "response": "x", "interaction_tokens": 30},
        {"id": "c", "error": None, "response": "x", "interaction_tokens": 20},
    ])
    st = rd.runtime_stats()
    check("runtime_stats reports the median interaction",
          st.get("interaction_tokens_median") == 20 and st.get("interaction_tokens_max") == 30,
          f"got {st}")

    import endpoint as ep
    from resultdir import no_answer_kind
    t0 = 0
    run = ep.finish_row({"id": "r"}, t0, response="", interaction_tokens=131072, max_tokens=131072)
    check("finish_row marks a cap-hit empty answer", run.get("no_answer") == "output_limit"
          and run["response"] == ep.NO_ANSWER_LIMIT and is_done(run), f"got {run}")
    run = ep.finish_row({"id": "r"}, t0, response="", interaction_tokens=130768, rounds=1)
    check("finish_row: an HLE loop near the cap without max_tokens is output_limit",
          run.get("no_answer") == "output_limit", f"got {run.get('no_answer')}")
    run = ep.finish_row({"id": "r"}, t0, response="", interaction_tokens=40)
    check("finish_row marks a short empty answer as empty", run.get("no_answer") == "empty")
    run = ep.finish_row({"id": "r"}, t0, response="", error="Timeout")
    check("finish_row leaves an error row unmarked",
          "no_answer" not in run and run["response"] == "" and not is_done(run))
    check("no_answer_kind reads an unmarked legacy runaway",
          no_answer_kind({"error": None, "response": "", "finish_reason": "length"}) == "output_limit")

    rd = ResultDir.start(tmp, suffix="na")
    write_rows(rd.responses_path, [
        {"id": "a", "error": None, "response": "x", "interaction_tokens": 100, "length": "0k"},
        {"id": "b", "error": None, "response": "x", "interaction_tokens": 300, "length": "0k"},
        {"id": "c", "error": None, "response": "x", "interaction_tokens": 5000, "length": "128k"},
        {"id": "d", "error": None, "response": ep.NO_ANSWER_LIMIT, "no_answer": "output_limit",
         "interaction_tokens": 131072, "length": "128k"},
        {"id": "e", "error": None, "response": "", "finish_reason": "length",
         "interaction_tokens": 131072, "length": "128k"},
        {"id": "f", "error": "boom", "response": "", "length": "128k"},
    ])
    st = rd.runtime_stats(stratify=lambda r: r.get("length"))
    check("runaways are left out of the interaction median and mean",
          st.get("interaction_tokens_n") == 3 and st.get("interaction_tokens_median") == 300
          and st.get("interaction_tokens_mean") == 1800.0, f"got {st}")
    check("runaways are counted, marked or not; errors are not",
          st.get("no_answer") == {"output_limit": 2, "empty": 0}
          and st.get("output_limit_rate") == 0.4, f"got {st.get('no_answer')} {st.get('output_limit_rate')}")
    by = st.get("interaction_by") or {}
    check("stratify reports each stratum separately",
          by.get("0k", {}).get("interaction_tokens_median") == 200
          and by.get("128k", {}).get("interaction_tokens_median") == 5000
          and by.get("128k", {}).get("no_answer", {}).get("output_limit") == 2, f"got {by}")


def test_ctxgate(tmp):
    """The HLE context gate: the budget, oldest first, an oversized item, runner wiring."""
    print("\nhle context gate: budget, priority, concurrency falling as contexts grow")
    import asyncio
    from types import SimpleNamespace as NS
    import ctxgate
    import runner

    async def grow(gate, log):
        slot = gate.slot()
        try:
            for r in range(1, 6):
                await slot.acquire(10 * r)
                log.append((r, gate.running, gate.held))
                await asyncio.sleep(0.002)
        finally:
            slot.release()

    async def four_growing():
        gate, log = ctxgate.ContextGate(100), []
        await asyncio.wait_for(asyncio.gather(*(grow(gate, log) for _ in range(4))), 10)
        return gate, log

    gate, log = asyncio.run(four_growing())
    check("ctx gate: the summed context never exceeds the budget",
          all(held <= 100 for _, _, held in log), f"got {log}")
    check("ctx gate: concurrency falls as contexts grow",
          max(n for r, n, _ in log if r == 1) == 4 and max(n for r, n, _ in log if r == 5) <= 2,
          f"got {log}")
    check("ctx gate: every item finishes and the gate drains",
          sum(1 for r, _, _ in log if r == 5) == 4 and (gate.held, gate.running) == (0, 0)
          and gate.pauses > 0, f"got {gate.__dict__}")

    async def oldest_first():
        gate, events = ctxgate.ContextGate(100), []
        a, b, c = gate.slot(), gate.slot(), gate.slot()
        await a.acquire(50)
        await b.acquire(40)

        async def apply(slot, name, need):
            await slot.acquire(need)
            events.append(name)

        ta = asyncio.create_task(apply(a, "a", 90))  # the oldest grows past what is free
        tc = asyncio.create_task(apply(c, "c", 5))   # the youngest would fit beside b
        await asyncio.sleep(0.01)
        blocked = (list(events), gate.held)
        tb = asyncio.create_task(apply(b, "b", 45))  # b re-applies, so it pauses
        await asyncio.sleep(0.01)
        mid = list(events)
        a.release()
        await asyncio.wait_for(asyncio.gather(ta, tb, tc), 1)
        b.release()
        c.release()
        return blocked, mid, events, gate.held

    blocked, mid, events, held = asyncio.run(oldest_first())
    check("ctx gate: a younger item waits behind a blocked older one", blocked == ([], 40),
          f"got {blocked}")
    check("ctx gate: a pausing item makes room for the older one; the rest follow by age",
          mid == ["a"] and events == ["a", "b", "c"] and held == 0,
          f"got {mid} then {events}, held {held}")

    async def oversized():
        gate = ctxgate.ContextGate(100)
        a, big = gate.slot(), gate.slot()
        await a.acquire(60)
        t = asyncio.create_task(big.acquire(500))
        await asyncio.sleep(0.01)
        waited = not t.done()
        a.release()
        await asyncio.wait_for(t, 1)
        alone = (gate.running, gate.held) == (1, 500)
        big.release()
        return waited, alone

    check("ctx gate: an item over the budget waits for the others, then runs alone",
          asyncio.run(oversized()) == (True, True))

    async def cancelled():
        gate = ctxgate.ContextGate(100)
        a, b, c = gate.slot(), gate.slot(), gate.slot()
        await a.acquire(80)
        tb = asyncio.create_task(b.acquire(50))
        tc = asyncio.create_task(c.acquire(10))
        await asyncio.sleep(0.01)
        waiting = not tc.done()
        tb.cancel()
        await asyncio.sleep(0.01)
        return waiting, tc.done(), gate.held

    check("ctx gate: a cancelled waiter steps aside", asyncio.run(cancelled()) == (True, True, 90))

    class Slow:
        """Round 1 calls a tool, round 2 answers; every request takes 50 ms."""
        def __init__(self):
            self.n = 0

        async def create(self, messages, **kw):
            self.n += 1
            n = self.n
            await asyncio.sleep(0.05)
            usage = NS(completion_tokens=50, model_dump=lambda: {"prompt_tokens": 100 * n,
                                                                 "completion_tokens": 50})
            calls = [NS(id="c1", function=NS(name="nope", arguments="{}"))] if n == 1 else None
            msg = NS(content=None if n == 1 else "42", tool_calls=calls)
            return NS(choices=[NS(message=msg)], usage=usage)

    async def wired():
        gate = ctxgate.ContextGate(200)
        rows = await asyncio.wait_for(asyncio.gather(
            runner.run_with_tools(Slow(), {"id": "p", "prompt": "q"}, ctx_gate=gate),
            runner.run_with_tools(Slow(), {"id": "q", "prompt": "q"}, ctx_gate=gate)), 10)
        return gate, rows

    gate, rows = asyncio.run(wired())
    check("hle: gated items both answer, one pausing while the other's context is held",
          [r.get("response") for r in rows] == ["42", "42"] and gate.pauses >= 1
          and any(r.get("ctx_pause_s", 0) > 0 for r in rows), f"got {rows}, {gate.__dict__}")
    check("hle: a finished gated run leaves nothing held",
          (gate.held, gate.running) == (0, 0), f"got {gate.__dict__}")
    row = asyncio.run(runner.run_with_tools(Slow(), {"id": "u", "prompt": "q"}))
    check("hle: an ungated row carries no ctx_pause_s",
          row.get("response") == "42" and "ctx_pause_s" not in row, f"got {row}")


def main():
    tmp = tempfile.mkdtemp(prefix="eai-selftest-")
    try:
        test_resume(tmp)
        test_config(tmp)
        test_grading(tmp)
        test_counting_stars(tmp)
        test_babilong_qa3(tmp)
        test_interaction(tmp)
        test_ctxgate(tmp)
        test_imported_roundtrip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
