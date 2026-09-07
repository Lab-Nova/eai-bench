#!/usr/bin/env python3
"""BFCL v4 benchmark client (500-task subset).

    setup   [--venv DIR]                       create the bfcl-eval virtualenv
    run     --endpoint URL --model NAME [--results-dir DIR] [--benchmark]
    collect --results-dir DIR                  bfcl evaluate -> score.json
    pending --results-dir DIR                  ids that still have no usable answer
    status  --results-dir DIR                  counts, per category
    latency [--results-dir DIR ...]            rank the subset by measured latency

BFCL is the odd component: it wraps the third-party `bfcl-eval` package instead of
talking to the endpoint directly, and it is machine-graded (AST + state), so there is
no judge skill and no `show`/`record`.

Everything for one run lives under <results-dir>/bfcl_root, which is passed to
bfcl-eval as BFCL_PROJECT_ROOT. That isolation is mandatory, not stylistic: `evaluate`
folds *every* score file it finds in its score directory into the leaderboard CSVs, so
two runs sharing a root silently blend into each other's numbers.

`run --benchmark` dispatches all 500 tasks in a fixed order at concurrency 256.
Generation has a shared five-minute deadline; unfinished tasks become timeout rows.
"""

import argparse
import csv
import glob
import json
import os
import multiprocessing
import signal
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import resultdir  # noqa: E402
import shim  # noqa: E402
import subset as subsetlib  # noqa: E402
from resultdir import ResultDir  # noqa: E402

COMPONENT = "bfcl"
CLIENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(CLIENT_ROOT, "results")
DEFAULT_VENV = os.path.join(CLIENT_ROOT, ".venv")
# Pin the version the current numbers were produced with. bfcl-eval's own `version`
# command warns that "pypi versions are in development, please rely on the commit hash
# for reproducibility", so `collect` records the installed version alongside the score.
BFCL_PIN = "bfcl-eval==2026.3.23"
# bfcl-eval 2026.3.23 does not declare these, but importing anything from
# bfcl_eval.constants.model_config pulls the whole handler registry, and the Qwen
# handler reaches qwen_agent -> qwen_agent.utils.utils -> `import soundfile`. Without
# it a clean install cannot even register a model:
#     ModuleNotFoundError: No module named 'soundfile'
# Installed alongside the pin rather than reported as a broken environment, because
# the venv is ours and the omission is upstream's.
BFCL_EXTRA_DEPS = ["soundfile"]
# Passed straight through to `bfcl generate --num-threads`. bfcl-eval's own default is
# 1 for an API handler (only OSS handlers get the concurrent default), which would run
# the whole subset serially.
DEFAULT_CONCURRENCY = 256
# Benchmark mode pins it instead of defaulting to it. Concurrency is the single biggest
# lever on both wall clock and per-request latency, so a benchmark run measured at any
# other value is not comparable with the rest -- and a run whose settings you have to go
# and check is not a benchmark. It is refused rather than overridden, so a --concurrency
# meant for a different run never passes silently. 256 is also the number the serving
# side is already sized for: serving/serve_glm5.3_dpep_conc128_hicache*.sh set
# --max-queued-requests 320 specifically to cover a client at this concurrency.
BENCHMARK_CONCURRENCY = 256
BENCHMARK_TIMEOUT_S = 300
# How many times `run` re-runs rows that came back holding an inference error. The same
# in every mode, benchmark included: a dropped request is the gateway's problem, not a
# different workload, and leaving it unanswered would put a wrong answer in the score
# and a missing request in the timing. The repair pass re-runs the same ids in the same
# relative order, so what it restores is the workload rather than perturbing it.
DEFAULT_MAX_ATTEMPTS = 3
# bfcl-eval's own default. It is not the sampling used by the other two components,
# which follow the model card's temperature 1.0 / top_p 0.95 -- recorded in config.json
# so the difference is visible rather than assumed.
DEFAULT_TEMPERATURE = 0.001

RESULT_GLOB = "BFCL_v4_*_result.json"


# -- venv ------------------------------------------------------------------------

def venv_python(venv):
    return os.path.join(venv, "bin", "python")


def ensure_venv(venv, quiet=False):
    py = venv_python(venv)
    if os.path.exists(py):
        return py
    print(f"creating virtualenv at {venv}", flush=True)
    subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"],
                   check=True, capture_output=quiet)
    print(f"installing {BFCL_PIN} (this pulls a few GB)", flush=True)
    subprocess.run([py, "-m", "pip", "install", BFCL_PIN, *BFCL_EXTRA_DEPS], check=True)
    return py


def reexec_in_venv(venv):
    """Re-run this script under the venv interpreter if bfcl_eval is not importable."""
    try:
        import bfcl_eval  # noqa: F401
        return
    except ImportError:
        pass
    py = venv_python(venv)
    if not os.path.exists(py):
        raise SystemExit(
            f"bfcl_eval is not importable and there is no venv at {venv}.\n"
            f"Run: python3 {sys.argv[0]} setup --venv {venv}")
    os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])


# -- result inspection -----------------------------------------------------------

def result_files(project_root, registry):
    return sorted(glob.glob(os.path.join(project_root, "result", registry, "**",
                                         RESULT_GLOB), recursive=True))


def category_of(path):
    return os.path.basename(path)[len("BFCL_v4_"):-len("_result.json")]


def is_failed(row):
    """True for a row bfcl-eval wrote in place of an answer it never got.

    `bfcl generate` catches every inference exception and writes
    `"Error during inference: <e>"` into the result field, with a valid id. On the next
    run that id counts as generated and is skipped forever, and `evaluate` grades it
    wrong. Upstream's reasoning -- retrying will not help at temperature 0.001 -- holds
    for a malformed model response and not at all for a connection error from a server
    restart, which is the failure that actually bit this project.
    """
    result = row.get("result")
    if isinstance(result, str) and result.startswith("Error during inference"):
        return True
    if "traceback" in row:
        return True
    if isinstance(result, list) and not any(result):
        return True
    return not isinstance(result, (str, list, dict)) or result == ""


def iter_rows(run_dir):
    """(category, row) for every result row under a *results* directory.

    Registry-agnostic on purpose: `latency` reads runs made against other endpoints and
    under other model names, and the registry only ever appears as one directory level.
    """
    pat = os.path.join(run_dir, "bfcl_root", "result", "*", "**", RESULT_GLOB)
    for path in sorted(glob.glob(pat, recursive=True)):
        cat = category_of(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield cat, json.loads(line)
                except json.JSONDecodeError:
                    # This reads other runs' files, including ones a killed process left
                    # with a torn final line. Everything before it is intact, and a
                    # latency ranking is not the place to die over one row.
                    continue


def _leaves(x):
    if isinstance(x, (int, float)):
        yield x
    elif isinstance(x, list):
        for y in x:
            yield from _leaves(y)


def row_latency(row):
    """Seconds this task held a worker thread, summed over every turn and every step.

    bfcl-eval records `latency` as a float for a single-turn task and as a list of lists
    -- turn, then step within the turn -- for a multi-turn one. The sum is the right
    number for both: one task occupies one thread from its first request to its last, so
    the total is what the scheduler sees and what the wall clock is made of.
    """
    return sum(_leaves(row.get("latency")))


def row_steps(row):
    """How many model calls the task took. One for single-turn, 9-30 for multi-turn."""
    return sum(1 for _ in _leaves(row.get("latency")))


def row_output_tokens(row):
    return sum(_leaves(row.get("output_token_count")))


def scan_results(project_root, registry):
    """-> (rows_by_category, failed_by_category)."""
    rows, failed = {}, {}
    for path in result_files(project_root, registry):
        cat = category_of(path)
        with open(path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        rows[cat] = entries
        bad = [e["id"] for e in entries if is_failed(e)]
        if bad:
            failed[cat] = bad
    return rows, failed


def write_id_file(project_root, grouped):
    path = os.path.join(project_root, "test_case_ids_to_generate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)
    return path


def _generation_child(argv, project_root, model, registry):
    os.setsid()
    shim.run_cli(argv, project_root, model, registry)


def run_generation_until(argv, project_root, model, registry, timeout_s):
    """Stop the entire generation process group at the shared wall-clock deadline."""
    process = multiprocessing.get_context("fork").Process(
        target=_generation_child, args=(argv, project_root, model, registry))
    process.start()
    try:
        process.join(max(0, timeout_s))
        if not process.is_alive():
            if process.exitcode:
                raise RuntimeError(f"bfcl generate exited {process.exitcode}")
            return True
        return False
    finally:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.join()


def mark_timeouts(project_root, registry, grouped):
    """After generation stops, preserve complete rows and mark missing IDs timeout.

    A killed writer can leave a partial last JSON line. Remove it before appending
    terminal rows, so both collection and later inspection can read every result.
    """
    have = set()
    paths = {}
    for path in result_files(project_root, registry):
        paths[category_of(path)] = path
        valid = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                have.add(row["id"])
                valid.append(row)
        with open(path, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row) + "\n")
    for cat, ids in grouped.items():
        path = paths.get(cat) or os.path.join(
            project_root, "result", registry, f"BFCL_v4_{cat}_result.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for task_id in ids:
                if task_id not in have:
                    f.write(json.dumps({"id": task_id, "status": "timeout",
                        "result": "Error during inference: timeout (benchmark 300s deadline)",
                        "timeout_s": BENCHMARK_TIMEOUT_S}) + "\n")


# -- commands --------------------------------------------------------------------

def cmd_setup(a):
    py = ensure_venv(a.venv)
    print(f"\nready: {py}")


def previous_benchmark(exclude_path=None):
    """(path, config) of the most recent earlier benchmark run, or (None, None)."""
    if not os.path.isdir(RESULTS_BASE):
        return None, None
    for name in sorted(os.listdir(RESULTS_BASE), reverse=True):
        path = os.path.join(RESULTS_BASE, name)
        if path == exclude_path or not os.path.isdir(path):
            continue
        cfg = ResultDir(path).config
        if cfg.get("benchmark"):
            return path, cfg
    return None, None


def cmd_run(a):
    reexec_in_venv(a.venv)

    # Benchmark mode settles its terms here, before anything is written, so that a run
    # either is the fixed workload or refuses to start pretending to be it.
    if a.benchmark:
        if a.target != subsetlib.TARGET:
            raise SystemExit("--benchmark requires --target 500")
        if a.results_dir:
            raise SystemExit(
                "--benchmark starts its own results directory and never resumes one.\n"
                "A resume generates only the ids still missing from the result files, "
                "which is a smaller workload in a different order -- the two things "
                "benchmark mode exists to hold fixed. Drop --results-dir; a half-"
                "finished benchmark run is discarded, not continued.")
        if a.concurrency not in (None, BENCHMARK_CONCURRENCY):
            raise SystemExit(
                f"--benchmark runs at concurrency {BENCHMARK_CONCURRENCY}, and "
                f"--concurrency {a.concurrency} would make this run incomparable with "
                f"every other benchmark run. Drop the flag, or drop --benchmark.")
        concurrency = BENCHMARK_CONCURRENCY
    else:
        concurrency = a.concurrency if a.concurrency is not None else DEFAULT_CONCURRENCY
    max_attempts = (a.max_attempts if a.max_attempts is not None
                    else DEFAULT_MAX_ATTEMPTS)

    if max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        items = rd.read_subset()
        if not items:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
    else:
        rd = ResultDir.start(RESULTS_BASE, suffix="bench" if a.benchmark else None)
        items = None  # built below, once BFCL_PROJECT_ROOT is set

    project_root = os.path.join(rd.path, "bfcl_root")
    os.makedirs(project_root, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = project_root
    registry = shim.registry_name(a.model)

    excluded = []
    if items is None:
        items = subsetlib.flatten(subsetlib.build(a.target))
        rd.write_subset(items, drop=())
    grouped = subsetlib.group(items)

    # The order bfcl-eval will actually dispatch in -- a pure function of the subset, so
    # it can be written down before a single request is sent and compared afterwards.
    order = subsetlib.dispatch_order(items)
    order_sha = subsetlib.order_fingerprint(order)
    if a.benchmark:
        rd._write_json(os.path.join(rd.path, "dispatch_order.json"), order)
        prev_path, prev_cfg = previous_benchmark(rd.path)
        if prev_cfg:
            prev_sha = prev_cfg.get("dispatch_order_sha256")
            verdict = "identical to" if prev_sha == order_sha else "DIFFERENT from"
            print(f"workload {verdict} the previous benchmark run "
                  f"{os.path.basename(prev_path)} ({prev_sha} -> {order_sha})")

    # Always regenerate the id file from subset.json: the repair pass narrows it, so it
    # is a scratch file and must never be treated as the source of truth.
    write_id_file(project_root, grouped)
    shim.write_env(project_root, a.endpoint, a.api_key)

    cfg = rd.reconcile({
        "component": COMPONENT, "mode": "benchmark" if a.benchmark else "native_fc",
        "endpoint": a.endpoint, "model": a.model, "registry_name": registry,
        "temperature": a.temperature, "n_subset": len(items),
        "concurrency": concurrency, "max_attempts": max_attempts,
        "bfcl_pin": BFCL_PIN,
        "started_at": rd.config.get("started_at") or resultdir.now_stamp(),
        "sampling_note": "bfcl-eval's own default temperature; the other two components "
                         "use the model card's temperature 1.0 / top_p 0.95",
        "benchmark": bool(a.benchmark),
        "long_tail_excluded": excluded,
        "generation_timeout_s": BENCHMARK_TIMEOUT_S if a.benchmark else None,
        "dispatch_order_sha256": order_sha,
    }, force=a.force)

    gen_argv = ["generate",
                "--model", registry,
                # Without --run-ids the id file is ignored entirely and bfcl runs the
                # full ~4.4k suite. The flag's help text says it *adds* to
                # --test-category; the code in fact replaces it.
                "--run-ids",
                "--num-threads", str(concurrency),
                "--temperature", str(a.temperature)]

    started, attempts_used = time.monotonic(), 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        print(f"\n=== generate (attempt {attempt}/{max_attempts}) ===", flush=True)
        if a.benchmark:
            remaining = BENCHMARK_TIMEOUT_S - (time.monotonic() - started)
            if remaining <= 0 or not run_generation_until(
                    gen_argv, project_root, cfg["model"], registry, remaining):
                mark_timeouts(project_root, registry, grouped)
                break
        else:
            shim.run_cli(gen_argv, project_root, cfg["model"], registry)

        _, failed = scan_results(project_root, registry)
        n_failed = sum(len(v) for v in failed.values())
        if not n_failed:
            print("no failed rows", flush=True)
            break
        print(f"{n_failed} row(s) hold an inference error; retrying just those",
              flush=True)
        if attempt == max_attempts:
            break
        # --run-ids together with --allow-overwrite is the one combination that leaves
        # the existing result file in place while regenerating exactly the listed ids,
        # and the write is an id-keyed upsert, so no duplicates appear.
        write_id_file(project_root, failed)
        gen_argv = ["generate", "--model", registry, "--run-ids", "--allow-overwrite",
                    "--num-threads", str(concurrency),
                    "--temperature", str(a.temperature)]
    wall = time.monotonic() - started

    write_id_file(project_root, grouped)  # restore the canonical list
    rows, failed = scan_results(project_root, registry)
    n_failed = sum(len(v) for v in failed.values())
    answered = sum(len(v) for v in rows.values()) - n_failed

    # One entry per invocation rather than one number: a resumed run is several waves of
    # requests and no single figure describes it. A benchmark run has exactly one.
    cfg = rd.config
    cfg.setdefault("generate_wall_s", []).append(round(wall, 1))
    cfg.setdefault("attempts_used", []).append(attempts_used)
    cfg["timeout_ids"] = {
        cat: [r["id"] for r in entries if r.get("status") == "timeout"]
        for cat, entries in rows.items()
        if any(r.get("status") == "timeout" for r in entries)
    }
    cfg["finished_at"] = resultdir.now_stamp()
    rd.write_config(cfg)

    print(f"\n{answered}/{len(items)} tasks answered in {wall:.0f}s")
    if a.benchmark:
        print_workload_summary(rd.path, wall, concurrency, attempts_used)
        if n_failed:
            print(f"\n{n_failed} task(s) have inference errors or timeouts; "
                  "these count as incorrect in the full subset denominator.")
    print(f"Next: python3 {__file__} collect --results-dir {rd.path}")


def print_workload_summary(run_dir, wall, concurrency, attempts_used=1):
    """What a benchmark run is for: the timing, next to the workload that produced it."""
    lat = sorted((row_latency(r), r["id"])
                 for _, r in iter_rows(run_dir) if not is_failed(r))
    if not lat:
        return
    vals = [x for x, _ in lat]
    total, n = sum(vals), len(vals)
    def q(p):
        return vals[min(n - 1, int(p * n))]
    print(f"\nworkload   : {n} tasks at concurrency {concurrency}")
    util = f"{total / (wall * concurrency) * 100:.1f}%" if wall > 0 else "n/a"
    print(f"wall clock : {wall:.0f}s   request-seconds {total:.0f}s   "
          f"utilisation {util} of {concurrency} slots")
    print(f"latency    : p50 {q(0.5):.1f}s  p90 {q(0.9):.1f}s  p99 {q(0.99):.1f}s  "
          f"max {vals[-1]:.1f}s")
    print("slowest    : " + ", ".join(f"{i} {v:.0f}s" for v, i in lat[-3:][::-1]))
    if attempts_used > 1:
        # The repair pass restores the workload, but its requests are inside the wall
        # clock: two runs are comparable on wall clock only if both took one pass.
        print(f"note       : took {attempts_used} generate passes -- the repair pass "
              f"re-ran dropped requests, and its time is inside the wall clock above")


def cmd_collect(a):
    if not a.no_evaluate:
        reexec_in_venv(a.venv)
    rd = ResultDir.open(a.results_dir)
    cfg = rd.config
    registry = cfg.get("registry_name") or shim.registry_name(cfg["model"])
    project_root = os.path.join(rd.path, "bfcl_root")
    os.environ["BFCL_PROJECT_ROOT"] = project_root

    if not a.no_evaluate:
        # --partial-eval is required: evaluate raises outright when the number of result
        # rows differs from the number of prompt entries, always true of a subset.
        shim.run_cli(["evaluate", "--model", registry, "--test-category", "all",
                      "--partial-eval"], project_root, cfg["model"], registry)

    breakdown, correct, total = {}, 0, 0
    for path in sorted(glob.glob(os.path.join(project_root, "score", registry, "**",
                                              "BFCL_v4_*_score.json"), recursive=True)):
        cat = os.path.basename(path)[len("BFCL_v4_"):-len("_score.json")]
        with open(path, encoding="utf-8") as f:
            head = json.loads(f.readline())
        breakdown[cat] = {"correct": head["correct_count"], "total": head["total_count"],
                          "accuracy": head["accuracy"]}
        correct += head["correct_count"]
        total += head["total_count"]

    n_subset = cfg.get("n_subset") or len(rd.read_subset())
    rows, failed = scan_results(project_root, registry)
    n_failed = sum(len(v) for v in failed.values())
    n_generated = sum(len(v) for v in rows.values())
    timeout_ids = {cat: [r["id"] for r in entries if r.get("status") == "timeout"]
                   for cat, entries in rows.items()
                   if any(r.get("status") == "timeout" for r in entries)}
    n_timeout = sum(map(len, timeout_ids.values()))

    score = {
        "component": COMPONENT,
        "correct": correct,
        # The denominator is the subset, always -- not the number bfcl managed to grade.
        "total": n_subset,
        "accuracy": round(correct / n_subset, 6) if n_subset else 0.0,
        "graded": total,
        "responded": n_generated - n_failed,
        "timeout_ids": timeout_ids,
        "timed_out": n_timeout,
        "benchmark_complete": bool(cfg.get("benchmark")) and n_generated == n_subset and total == n_subset and n_failed == n_timeout,
        "complete": total == n_subset and n_failed == 0,
        "grader": "bfcl-eval:ast+state",
        "score_definition": "flat pass rate = correct / n_subset",
        "by_category": breakdown,
        "failed_ids": failed,
        "results_dir": rd.path,
        "scored_at": resultdir.now_stamp(),
    }
    # Deadline-limited benchmark scores remain separate from uncapped accuracy.
    for k in ("endpoint", "model", "registry_name", "temperature", "imported",
              "mode", "benchmark", "long_tail_excluded", "dispatch_order_sha256",
              "concurrency", "generate_wall_s", "generation_timeout_s", "attempts_used"):
        if k in cfg:
            score[k] = cfg[k]

    # bfcl-eval's own leaderboard number is category-weighted and treats the categories
    # we did not run as N/A. It is not comparable with the other components and is never
    # summed into the composite -- carried only as a labelled footnote.
    overall_csv = os.path.join(project_root, "score", "data_overall.csv")
    if os.path.exists(overall_csv):
        with open(overall_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            score["native_weighted_overall"] = {
                "overall_acc": rows[0].get("Overall Acc"),
                "latency_mean_s": rows[0].get("Latency Mean (s)"),
                "note": "bfcl-eval's category-weighted leaderboard formula; counts "
                        "un-run categories as N/A, not comparable across components"}

    rd._write_json(rd.score_path, score)
    print(json.dumps({k: v for k, v in score.items() if k != "by_category"}, indent=2))
    if cfg.get("benchmark"):
        print(f"\nbenchmark run: {n_subset} tasks, {n_timeout} timeouts; "
              "timeouts count as incorrect. synthesis.py skips benchmark scores.")
    print("\nby category:")
    for cat, b in sorted(breakdown.items()):
        print(f"  {cat:28s} {b['correct']:4d}/{b['total']:<4d} {b['accuracy']:.4f}")


def cmd_pending(a):
    rd = ResultDir.open(a.results_dir)
    cfg = rd.config
    registry = cfg.get("registry_name") or shim.registry_name(cfg["model"])
    project_root = os.path.join(rd.path, "bfcl_root")
    rows, failed = scan_results(project_root, registry)
    have = {e["id"] for entries in rows.values() for e in entries if not is_failed(e)}
    missing = [it["id"] for it in rd.read_subset() if it["id"] not in have]
    print(json.dumps(missing, indent=2))


def cmd_status(a):
    rd = ResultDir.open(a.results_dir)
    cfg = rd.config
    registry = cfg.get("registry_name") or shim.registry_name(cfg["model"])
    project_root = os.path.join(rd.path, "bfcl_root")
    rows, failed = scan_results(project_root, registry)
    items = rd.read_subset()
    print(f"results dir : {rd.path}")
    print(f"endpoint    : {cfg.get('endpoint')}  model: {cfg.get('model')}")
    print(f"subset      : {len(items)} tasks in {len(subsetlib.group(items))} categories")
    if cfg.get("benchmark"):
        print(f"benchmark   : concurrency {cfg.get('concurrency')}  order "
              f"{cfg.get('dispatch_order_sha256')}  wall "
              f"{', '.join(f'{w:.0f}s' for w in cfg.get('generate_wall_s') or [])}")
        print(f"  excluded  : {', '.join(cfg.get('long_tail_excluded') or []) or 'none'}")
    generated = sum(len(v) for v in rows.values())
    n_failed = sum(len(v) for v in failed.values())
    print(f"generated   : {generated}  failed rows: {n_failed}")
    for cat, ids in sorted(failed.items()):
        print(f"  {cat:28s} {len(ids)} failed")


def cmd_latency(a):
    """Rank the subset by measured per-request latency -- where LONG_TAIL comes from.

    The measurement lives in the result rows bfcl-eval already writes, so this needs no
    endpoint, no venv and no re-run: it re-derives the ranking from whatever finished
    runs are on disk. Aggregating across runs is the point -- in a single run the
    ranking is as much a picture of the endpoint's bad minutes as of the suite, and one
    task here moved from 5 s to 1088 s between two runs.

    `fail` is part of the ranking, not a footnote to it. A failed row carries no latency
    and so drops out of the median, which quietly flatters exactly the tasks that are
    slow enough to hit a gateway timeout: the worst task in this suite is missing from
    half the runs for that reason, and would otherwise look like it appears rarely
    rather than like it falls over.
    """
    dirs = a.results_dir or [os.path.join(RESULTS_BASE, n)
                             for n in sorted(os.listdir(RESULTS_BASE))
                             if os.path.isdir(os.path.join(RESULTS_BASE, n))]
    per_id, used = {}, []
    for d in dirs:
        rows = list(iter_rows(d))
        good = [(c, r) for c, r in rows if not is_failed(r)]
        # Smoke runs of a couple of dozen tasks would otherwise contribute a median over
        # a handful of categories and skew every id they happen to contain.
        if len(good) < a.min_rows:
            continue
        used.append((d, len(good), len(rows) - len(good)))
        for cat, r in rows:
            e = per_id.setdefault(r["id"], {"cat": cat, "lat": [], "steps": [],
                                            "out": [], "fail": 0})
            if is_failed(r):
                e["fail"] += 1
                continue
            e["lat"].append(row_latency(r))
            e["steps"].append(row_steps(r))
            e["out"].append(row_output_tokens(r))
    if not used:
        raise SystemExit(f"no run under {RESULTS_BASE} has {a.min_rows}+ usable rows "
                         f"(pass --min-rows, or --results-dir)")

    print(f"{len(used)} run(s), {len(per_id)} task(s):")
    for d, n, nf in used:
        print(f"  {os.path.basename(d):28s} {n:4d} rows"
              + (f"  ({nf} failed)" if nf else ""))

    ranked = sorted(((statistics.median(e["lat"]), min(e["lat"]), max(e["lat"]),
                      statistics.median(e["steps"]), statistics.median(e["out"]),
                      len(e["lat"]), e["fail"], i)
                     for i, e in per_id.items() if len(e["lat"]) >= a.min_runs),
                    reverse=True)
    if not ranked:
        raise SystemExit(f"no task appears in {a.min_runs}+ of those runs")
    total = sum(r[0] for r in ranked)
    cut = len(subsetlib.LONG_TAIL)

    print(f"\nslowest {a.top}, by median total latency over the runs that answered them:")
    print(f"  {'med':>8} {'min':>8} {'max':>8} {'runs':>5} {'fail':>4} {'steps':>6} "
          f"{'out_tok':>8}  task")
    for med, lo, hi, steps, out, n, nfail, i in ranked[:a.top]:
        mark = " *" if i in subsetlib.LONG_TAIL else "  "
        print(f"{mark}{med:8.1f} {lo:8.1f} {hi:8.1f} {n:5d} {nfail:4d} {steps:6.0f} "
              f"{out:8.0f}  {i}")
    print("\n* = in subset.LONG_TAIL, the set `run --benchmark` excludes.")
    print(f"  the top {cut} are {sum(r[0] for r in ranked[:cut]) / total * 100:.1f}% of "
          f"the {total:.0f} median request-seconds in a run.")
    drift = [r[-1] for r in ranked[:cut] if r[-1] not in subsetlib.LONG_TAIL]
    if drift:
        print(f"  DRIFT: the measured top {cut} no longer matches LONG_TAIL. "
              f"Not in it: {', '.join(drift)}.")
        print("  Changing LONG_TAIL changes the workload, so it is a deliberate edit to "
              "subset.py, not something this command does.")


def cmd_refuse(a):
    raise SystemExit(
        "bfcl is machine-graded by bfcl-eval (AST + state); there are no manual "
        "verdicts to show or record. Use `collect` to produce score.json.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venv", default=DEFAULT_VENV, help=f"default {DEFAULT_VENV}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="create the bfcl-eval virtualenv")
    s.set_defaults(fn=cmd_setup)

    r = sub.add_parser("run", help="run or resume generation")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True, help="the server's --served-model-name")
    r.add_argument("--results-dir", help="resume into this directory instead of creating one")
    r.add_argument("--target", type=int, default=subsetlib.TARGET)
    r.add_argument("--concurrency", type=int, default=None,
                   help=f"default {DEFAULT_CONCURRENCY}; fixed at "
                        f"{BENCHMARK_CONCURRENCY} under --benchmark")
    r.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    r.add_argument("--api-key", default="EMPTY")
    r.add_argument("--max-attempts", type=int, default=None,
                   help=f"how many times to retry rows holding an inference error "
                        f"(default {DEFAULT_MAX_ATTEMPTS}, every mode)")
    r.add_argument("--benchmark", action="store_true",
                   help="500 tasks with a five-minute generation deadline, "
                        f"one dispatch order, concurrency "
                        f"{BENCHMARK_CONCURRENCY}, no resume; synthesis.py skips it.")
    r.add_argument("--force", action="store_true",
                   help="resume even though endpoint/model changed (recorded in config.json)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("collect", help="evaluate and write score.json")
    s.add_argument("--results-dir", required=True)
    s.add_argument("--no-evaluate", action="store_true",
                   help="fold the score files already present instead of re-running "
                        "bfcl evaluate; needs no venv")
    s.set_defaults(fn=cmd_collect)

    for name, fn, help_ in (("pending", cmd_pending, "ids with no usable answer"),
                            ("status", cmd_status, "run summary")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--results-dir", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("latency", help="rank the subset by measured latency")
    s.add_argument("--results-dir", nargs="*",
                   help="runs to measure over (default: every run under results/)")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--min-rows", type=int, default=100,
                   help="ignore runs with fewer usable rows than this (default 100)")
    s.add_argument("--min-runs", type=int, default=3,
                   help="ignore tasks measured in fewer runs than this (default 3)")
    s.set_defaults(fn=cmd_latency)

    for name in ("show", "record"):
        s = sub.add_parser(name, help="not applicable: bfcl is machine-graded")
        s.add_argument("--results-dir")
        s.add_argument("rest", nargs="*")
        s.set_defaults(fn=cmd_refuse)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
