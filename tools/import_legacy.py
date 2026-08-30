#!/usr/bin/env python3
"""Import the original eai-v1.0 run into the new results-directory layout.

One-shot: the old run is a pile of flat files beside the code
(`responses_hle.jsonl`, `grades_hle_tools.d/`, `bfcl/score/...`). This converts it into
four dated result directories -- HLE has two passes, so two directories -- so the
historical numbers live under the same contract as anything run from now on, and
`synthesis.py` can reproduce them.

    python3 tools/import_legacy.py [--source ~/workspace-hle/eai-v1.0] [--dry-run]

The imported directories are marked `"imported": true` in config.json, with the config
drift recorded: BFCL and AA-LCR ran on b200t2 at concurrency 128 / context 163840, and
HLE with-tools on b200t3 at concurrency 256 / context 655360.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_SOURCE = os.path.expanduser("~/workspace-hle/eai-v1.0")

ENDPOINT = "http://127.0.0.1:30000/v1"
MODEL = "GLM-5.3"

# The two hosts the run spanned, reconstructed from the launch scripts and run logs.
B200T2 = {"host": "b200t2", "max_running_requests": 128, "context_length": 163840,
          "mem_fraction_static": 0.85, "hicache_ratio": 8}
B200T3 = {"host": "b200t3", "max_running_requests": 256, "context_length": 655360,
          "mem_fraction_static": 0.85, "hicache_ratio": 8}


def stamp_of(path):
    """Use the artifact's own mtime, so imported runs sort chronologically."""
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime(os.path.getmtime(path)))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def import_jsonl_component(src, dest_base, *, responses, subset, grades_dir, config,
                           suffix=None, dry_run=False):
    """HLE and AA-LCR: copy responses, subset and the per-item grade files."""
    resp_src = os.path.join(src, responses)
    if not os.path.exists(resp_src):
        print(f"  SKIP {responses}: not present")
        return None
    ts = stamp_of(resp_src) + (f"-{suffix}" if suffix else "")
    dest = os.path.join(dest_base, ts)
    print(f"  {responses} -> {os.path.relpath(dest, REPO)}")
    if dry_run:
        return dest
    os.makedirs(os.path.join(dest, "grades"), exist_ok=True)
    shutil.copy2(resp_src, os.path.join(dest, "responses.jsonl"))
    shutil.copy2(os.path.join(src, subset), os.path.join(dest, "subset.json"))
    n_grades = 0
    gsrc = os.path.join(src, grades_dir)
    if os.path.isdir(gsrc):
        for name in os.listdir(gsrc):
            if name.endswith(".json"):
                shutil.copy2(os.path.join(gsrc, name),
                             os.path.join(dest, "grades", name))
                n_grades += 1
    write_json(os.path.join(dest, "config.json"), config)
    print(f"    {n_grades} grade files")
    return dest


def import_bfcl(src, dest_base, dry_run=False):
    """BFCL: the whole bfcl-eval project root moves under the results directory."""
    broot = os.path.join(src, "bfcl")
    id_file = os.path.join(broot, "test_case_ids_to_generate.json")
    if not os.path.exists(id_file):
        print("  SKIP bfcl: no test_case_ids_to_generate.json")
        return None
    ts = stamp_of(id_file)
    dest = os.path.join(dest_base, ts)
    print(f"  bfcl/ -> {os.path.relpath(dest, REPO)}")
    if dry_run:
        return dest
    os.makedirs(dest, exist_ok=True)
    project_root = os.path.join(dest, "bfcl_root")
    if os.path.isdir(project_root):
        shutil.rmtree(project_root)
    # .file_locks is scratch state from the original run and carries no information.
    shutil.copytree(broot, project_root,
                    ignore=shutil.ignore_patterns(".file_locks"))

    with open(id_file, encoding="utf-8") as f:
        grouped = json.load(f)
    items = [{"id": i, "category": c} for c in sorted(grouped) for i in grouped[c]]
    write_json(os.path.join(dest, "subset.json"), items)
    write_json(os.path.join(dest, "config.json"), {
        "component": "bfcl", "mode": "native_fc", "imported": True,
        "endpoint": ENDPOINT, "model": MODEL,
        "registry_name": "GLM-5.3-sglang-FC",
        "temperature": 0.001, "n_subset": len(items),
        "concurrency": 32, "max_attempts": 1,
        "bfcl_pin": "bfcl-eval==2026.3.23",
        "serving": B200T2,
        "note": "imported from eai-v1.0; --num-threads was not recorded and 32 is a "
                "reconstruction from overlapping spans in bfcl_gen.log (>=30)",
    })
    print(f"    {len(items)} subset ids")
    return dest


def run(cmd):
    print(f"  $ {' '.join(cmd[-4:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    src = os.path.expanduser(a.source)
    if not os.path.isdir(src):
        raise SystemExit(f"no such source directory: {src}")

    common = {"endpoint": ENDPOINT, "model": MODEL, "imported": True,
              "temperature": 1.0, "top_p": 0.95}

    print("HLE (no tools):")
    hle_nt = import_jsonl_component(
        src, os.path.join(REPO, "hle", "results"),
        responses="responses_hle.jsonl", subset="subset_hle.json",
        grades_dir="grades_hle.d", suffix="notools", dry_run=a.dry_run,
        config={**common, "component": "hle", "mode": "no_tools",
                "n_subset": 250, "concurrency": 64, "max_tokens": 65536,
                "max_rounds": 0, "serving": B200T2,
                "note": "imported from eai-v1.0; the 65536 cap truncated 36 of 250 "
                        "items to empty answers, and all 36 scored zero"})

    print("HLE (with tools):")
    hle_t = import_jsonl_component(
        src, os.path.join(REPO, "hle", "results"),
        # Both passes cover the identical 250 ids -- subset_hle.json and
        # subset_hle_tools.json are byte-identical -- so mode is the discriminator.
        responses="responses_hle_tools.jsonl", subset="subset_hle.json",
        grades_dir="grades_hle_tools.d", suffix="tools", dry_run=a.dry_run,
        config={**common, "component": "hle", "mode": "tools",
                "n_subset": 250, "concurrency": 256, "max_tokens": 0,
                "max_rounds": 0, "serving": B200T3,
                "note": "imported from eai-v1.0; rows 1-63 were generated on b200t2 at "
                        "concurrency 128 / context 1048576 before the run moved to "
                        "b200t3 at concurrency 256 / context 655360"})

    print("AA-LCR:")
    aalcr = import_jsonl_component(
        src, os.path.join(REPO, "aalcr", "results"),
        responses="responses_aalcr.jsonl", subset="subset_aalcr.json",
        grades_dir="grades_aalcr.d", dry_run=a.dry_run,
        config={**common, "component": "aalcr", "mode": "single_turn",
                "n_subset": 25, "concurrency": 8, "max_tokens": 24576,
                "serving": B200T2, "note": "imported from eai-v1.0"})

    print("BFCL:")
    bfcl = import_bfcl(src, os.path.join(REPO, "bfcl", "results"), dry_run=a.dry_run)

    if a.dry_run:
        print("\ndry run; nothing written")
        return

    print("\nscoring:")
    for path, client, extra in ((hle_nt, "hle", []), (hle_t, "hle", []),
                                (aalcr, "aalcr", []),
                                (bfcl, "bfcl", ["--no-evaluate"])):
        if path is None:
            continue
        out = run([sys.executable, os.path.join(REPO, client, "src", "main.py"),
                   "collect", "--results-dir", path, *extra])
        score = json.loads(out[:out.index("\n}") + 2]) if out.startswith("{") else None
        if score:
            print(f"  {client:6s} {score['correct']:4d}/{score['total']:<4d} "
                  f"{100 * score['accuracy']:6.2f}%  "
                  f"{'complete' if score['complete'] else 'INCOMPLETE'}")

    print("\nnow run: python3 synthesis.py")


if __name__ == "__main__":
    main()
