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


def main():
    tmp = tempfile.mkdtemp(prefix="eai-selftest-")
    try:
        test_resume(tmp)
        test_config(tmp)
        test_grading(tmp)
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
