#!/usr/bin/env python3
"""BFCL v4 benchmark client (500-task subset).

    setup   [--venv DIR]                       create the bfcl-eval virtualenv
    run     --endpoint URL --model NAME [--results-dir DIR]
    collect --results-dir DIR                  bfcl evaluate -> score.json
    pending --results-dir DIR                  ids that still have no usable answer
    status  --results-dir DIR                  counts, per category

BFCL is the odd component: it wraps the third-party `bfcl-eval` package instead of
talking to the endpoint directly, and it is machine-graded (AST + state), so there is
no judge skill and no `show`/`record`.

Everything for one run lives under <results-dir>/bfcl_root, which is passed to
bfcl-eval as BFCL_PROJECT_ROOT. That isolation is mandatory, not stylistic: `evaluate`
folds *every* score file it finds in its score directory into the leaderboard CSVs, so
two runs sharing a root silently blend into each other's numbers.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

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
# The default is 1 for an API handler (only OSS handlers get the concurrent default),
# which would run the whole subset serially.
DEFAULT_CONCURRENCY = 32
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
    subprocess.run([py, "-m", "pip", "install", BFCL_PIN], check=True)
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


# -- commands --------------------------------------------------------------------

def cmd_setup(a):
    py = ensure_venv(a.venv)
    print(f"\nready: {py}")


def cmd_run(a):
    reexec_in_venv(a.venv)

    if a.results_dir:
        rd = ResultDir.open(a.results_dir)
        items = rd.read_subset()
        if not items:
            raise SystemExit(f"{rd.path} has no subset.json -- it is not a run directory")
    else:
        rd = ResultDir.start(RESULTS_BASE)
        items = None  # built below, once BFCL_PROJECT_ROOT is set

    project_root = os.path.join(rd.path, "bfcl_root")
    os.makedirs(project_root, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = project_root
    registry = shim.registry_name(a.model)

    if items is None:
        items = subsetlib.flatten(subsetlib.build(a.target))
        rd.write_subset(items, drop=())
    grouped = subsetlib.group(items)

    # Always regenerate the id file from subset.json: the repair pass narrows it, so it
    # is a scratch file and must never be treated as the source of truth.
    write_id_file(project_root, grouped)
    shim.write_env(project_root, a.endpoint, a.api_key)

    cfg = rd.reconcile({
        "component": COMPONENT, "mode": "native_fc",
        "endpoint": a.endpoint, "model": a.model, "registry_name": registry,
        "temperature": a.temperature, "n_subset": len(items),
        "concurrency": a.concurrency, "max_attempts": a.max_attempts,
        "bfcl_pin": BFCL_PIN,
        "started_at": rd.config.get("started_at") or resultdir.now_stamp(),
        "sampling_note": "bfcl-eval's own default temperature; the other two components "
                         "use the model card's temperature 1.0 / top_p 0.95",
    }, force=a.force)

    gen_argv = ["generate",
                "--model", registry,
                # Without --run-ids the id file is ignored entirely and bfcl runs the
                # full ~4.4k suite. The flag's help text says it *adds* to
                # --test-category; the code in fact replaces it.
                "--run-ids",
                "--num-threads", str(a.concurrency),
                "--temperature", str(a.temperature)]

    for attempt in range(1, a.max_attempts + 1):
        print(f"\n=== generate (attempt {attempt}/{a.max_attempts}) ===", flush=True)
        shim.run_cli(gen_argv, project_root, cfg["model"], registry)

        _, failed = scan_results(project_root, registry)
        n_failed = sum(len(v) for v in failed.values())
        if not n_failed:
            print("no failed rows", flush=True)
            break
        print(f"{n_failed} row(s) hold an inference error; retrying just those",
              flush=True)
        if attempt == a.max_attempts:
            break
        # --run-ids together with --allow-overwrite is the one combination that leaves
        # the existing result file in place while regenerating exactly the listed ids,
        # and the write is an id-keyed upsert, so no duplicates appear.
        write_id_file(project_root, failed)
        gen_argv = ["generate", "--model", registry, "--run-ids", "--allow-overwrite",
                    "--num-threads", str(a.concurrency),
                    "--temperature", str(a.temperature)]

    write_id_file(project_root, grouped)  # restore the canonical list
    rows, failed = scan_results(project_root, registry)
    answered = sum(len(v) for v in rows.values()) - sum(len(v) for v in failed.values())
    print(f"\n{answered}/{len(items)} tasks answered. Next: "
          f"python3 {__file__} collect --results-dir {rd.path}")


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
    _, failed = scan_results(project_root, registry)
    n_failed = sum(len(v) for v in failed.values())

    score = {
        "component": COMPONENT,
        "correct": correct,
        # The denominator is the subset, always -- not the number bfcl managed to grade.
        "total": n_subset,
        "accuracy": round(correct / n_subset, 6) if n_subset else 0.0,
        "graded": total,
        "responded": n_subset - n_failed,
        "complete": total == n_subset and n_failed == 0,
        "grader": "bfcl-eval:ast+state",
        "score_definition": "flat pass rate = correct / n_subset",
        "by_category": breakdown,
        "failed_ids": failed,
        "results_dir": rd.path,
        "scored_at": resultdir.now_stamp(),
    }
    for k in ("endpoint", "model", "registry_name", "temperature", "imported"):
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
    generated = sum(len(v) for v in rows.values())
    n_failed = sum(len(v) for v in failed.values())
    print(f"generated   : {generated}  failed rows: {n_failed}")
    for cat, ids in sorted(failed.items()):
        print(f"  {cat:28s} {len(ids)} failed")


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
    r.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    r.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    r.add_argument("--api-key", default="EMPTY")
    r.add_argument("--max-attempts", type=int, default=3,
                   help="how many times to retry rows holding an inference error")
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

    for name in ("show", "record"):
        s = sub.add_parser(name, help="not applicable: bfcl is machine-graded")
        s.add_argument("--results-dir")
        s.add_argument("rest", nargs="*")
        s.set_defaults(fn=cmd_refuse)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
