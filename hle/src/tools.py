"""The two tools the model gets during an HLE with-tools run: Python and web search.

Both executors are synchronous and are called from the async runner through a thread
pool, so a slow subprocess or HTTP fetch never blocks the event loop.

Tool results are truncated to TOOL_OUTPUT_LIMIT characters. HLE answers hinge on a
computed value or a looked-up fact, not on bulk text, and an untruncated dump can
crowd the reasoning budget out of the context window.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

TOOL_OUTPUT_LIMIT = 8000
PYTHON_TIMEOUT = int(os.environ.get("HLE_PY_TIMEOUT", "120"))
# Model-written code runs unsupervised, and a single unbounded allocation (a sieve sized
# 10**10, say) can take the whole box down with it -- that is exactly what OOM-killed the
# container once. Every child therefore gets a hard address-space and file-size ceiling.
PYTHON_MEM_LIMIT_GB = float(os.environ.get("HLE_PY_MEM_GB", "8"))
PY_WORKERS = int(os.environ.get("HLE_PY_WORKERS", "16"))
SEARCH_WORKERS = int(os.environ.get("HLE_SEARCH_WORKERS", "8"))
UA = "Mozilla/5.0 (compatible; EAI-eval/1.0; +https://artificialanalysis.ai)"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python 3 code and return its stdout and stderr. The process has "
                "internet access and the standard scientific stack (numpy, scipy, sympy, "
                "pandas). State is NOT preserved between calls, so each call must be a "
                "complete program. Print anything you want to see - values are not echoed "
                "automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python 3 source to run."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web and return ranked results as title, URL and snippet. Use it "
                "to look up facts, definitions, papers and data you are unsure about. To read "
                "a full page, fetch the URL with the python tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _clip(s):
    if len(s) <= TOOL_OUTPUT_LIMIT:
        return s
    half = TOOL_OUTPUT_LIMIT // 2
    return f"{s[:half]}\n...[{len(s) - TOOL_OUTPUT_LIMIT} characters truncated]...\n{s[-half:]}"


def run_python(code):
    """Run `code` in a throwaway working directory and report both streams."""
    with tempfile.TemporaryDirectory(prefix="hle_py_") as cwd:
        # The code goes in its own file and a wrapper applies the rlimits before running
        # it. Setting them in the parent via preexec_fn would not be thread-safe, and the
        # wrapper keeps tracebacks pointing at real main.py line numbers.
        with open(os.path.join(cwd, "main.py"), "w") as fh:
            fh.write(code)
        with open(os.path.join(cwd, "_runner.py"), "w") as fh:
            fh.write(
                "import os, resource, runpy, sys\n"
                # RLIMIT_AS caps address space, not resident memory, and OpenBLAS reserves
                # a stack per thread -- 64 of them overrun any sane ceiling before a single
                # array is allocated. Pinning the thread count keeps numpy/scipy usable and
                # stops 64 concurrent children from each grabbing every core.
                "for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',\n"
                "           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):\n"
                "    os.environ[_v] = '4'\n"
                f"lim = int({PYTHON_MEM_LIMIT_GB} * 1024**3)\n"
                "resource.setrlimit(resource.RLIMIT_AS, (lim, lim))\n"
                "resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)\n"
                "runpy.run_path(sys.argv[1], run_name='__main__')\n"
            )
        try:
            p = subprocess.run(
                [sys.executable, "_runner.py", "main.py"],
                capture_output=True, text=True, timeout=PYTHON_TIMEOUT, cwd=cwd,
            )
            out = p.stdout or ""
            err = p.stderr or ""
            parts = []
            if out.strip():
                parts.append(f"STDOUT:\n{out}")
            if err.strip():
                parts.append(f"STDERR:\n{err}")
            if not parts:
                parts.append("(no output - remember to print() what you want to see)")
            if p.returncode != 0:
                parts.append(f"(exit code {p.returncode})")
            if "MemoryError" in err or p.returncode == -9:
                parts.append(f"(the process is limited to {PYTHON_MEM_LIMIT_GB} GB of memory; "
                             f"use a smaller or more memory-efficient computation)")
            return _clip("\n".join(parts))
        except subprocess.TimeoutExpired:
            return f"ERROR: execution exceeded {PYTHON_TIMEOUT}s and was killed."
        except Exception as e:  # noqa: BLE001 - returned to the model as tool output
            return f"ERROR: {type(e).__name__}: {e}"


def _wikipedia(query, n):
    api = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
           f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit={n}")
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.load(r)
    out = []
    for s in data.get("query", {}).get("search", []):
        title = s["title"]
        snippet = (s.get("snippet", "").replace('<span class="searchmatch">', "")
                   .replace("</span>", ""))
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        out.append({"title": title, "href": url, "body": snippet})
    return out


def web_search(query, max_results=5):
    """DuckDuckGo first; fall back to the Wikipedia search API if it yields nothing."""
    n = max(1, min(int(max_results or 5), 10))
    results, errors = [], []
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=n))
    except Exception as e:  # noqa: BLE001 - reported to the model, then fall back
        errors.append(f"duckduckgo: {type(e).__name__}: {e}")
    if not results:
        try:
            results = _wikipedia(query, n)
        except Exception as e:  # noqa: BLE001
            errors.append(f"wikipedia: {type(e).__name__}: {e}")
    if not results:
        return "No results. " + ("; ".join(errors) if errors else "")
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title','')}\n   {r.get('href','')}\n   {r.get('body','')}")
    return _clip("\n".join(lines))


# Python execution is CPU-bound and search is network-bound, so they get separate pools:
# a burst of searches must not queue behind long-running code, and DuckDuckGo starts
# refusing requests if too many land at once. Both are built on first use -- importing
# this module must not spawn 24 threads in a process that only wanted TOOLS.
_POOLS = {}


def _pool(kind, workers):
    if kind not in _POOLS:
        _POOLS[kind] = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"hle-{kind}")
    return _POOLS[kind]


def shutdown():
    """Release the tool pools; safe to call when no tool call is in flight."""
    while _POOLS:
        _, pool = _POOLS.popitem()
        pool.shutdown(wait=False)


def dispatch(name, arguments):
    """Map an OpenAI tool call to (pool, callable) so the runner can await it."""
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return None, (lambda: f"ERROR: arguments were not valid JSON: {e}")
    if name == "python":
        return _pool("py", PY_WORKERS), (lambda: run_python(args.get("code", "")))
    if name == "web_search":
        return (_pool("search", SEARCH_WORKERS),
                lambda: web_search(args.get("query", ""), args.get("max_results", 5)))
    return None, (lambda: f"ERROR: no such tool '{name}'")
