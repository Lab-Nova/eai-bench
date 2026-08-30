"""Loading `cais/hle` and choosing the 250-question subset.

The subset is chosen without an RNG: sort the text-only rows by id and take an even
stride. The same code therefore produces the same 250 questions on any machine, in
either mode, which is what makes the with-tools and no-tools passes comparable.
"""

import os

N_SUBSET = 250

SYSTEM_EXACT_ANSWER = """Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

SYSTEM_MC = """Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

TOOL_PREAMBLE = """You have two tools available: `python` (run Python 3 code, returns stdout/stderr) and `web_search` (search the web). Use them whenever calculation or a factual lookup would make your answer more reliable. When you are done using tools, give your final answer in the required format."""

GATED_HELP = """\
`cais/hle` is a gated dataset. Either:
  * export HF_TOKEN=<token from an account that accepted the terms on
    https://huggingface.co/datasets/cais/hle>, or
  * pre-populate the HF cache and run offline:
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 hle/src/main.py run ...
"""


def load_subset(n_subset=N_SUBSET, with_tools=True, limit=None):
    """Return the subset as items with an `id`, metadata, gold answer and `prompt`."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover - environment problem, not a run problem
        raise SystemExit("the `datasets` package is required: pip install datasets")

    offline = os.environ.get("HF_DATASETS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not offline:
        # Fail here rather than several minutes into a run: the 401 is about access,
        # and nothing about it improves by discovering it later.
        raise SystemExit(GATED_HELP)
    try:
        ds = load_dataset("cais/hle", split="test", token=token)
    except Exception as e:  # noqa: BLE001 - turn the SDK's 401 into the actionable message
        raise SystemExit(f"could not load cais/hle: {type(e).__name__}: {e}\n\n{GATED_HELP}")

    # Image questions are excluded outright, so they never enter the denominator.
    text_only = [r for r in ds if not r.get("image")]
    print(f"excluded {len(ds) - len(text_only)} image questions; "
          f"{len(text_only)} text-only of {len(ds)}", flush=True)

    text_only.sort(key=lambda r: str(r["id"]))
    n = len(text_only)
    if n_subset > n:
        raise SystemExit(f"asked for {n_subset} questions but only {n} are text-only")
    subset = [text_only[j * n // n_subset] for j in range(n_subset)]
    assert not any(r.get("image") for r in subset), "an image question reached the subset"
    if limit:
        subset = subset[:limit]

    items = []
    for r in subset:
        system = SYSTEM_MC if r.get("answer_type") == "multipleChoice" else SYSTEM_EXACT_ANSWER
        preamble = f"{system}\n\n{TOOL_PREAMBLE}" if with_tools else system
        items.append({
            "id": str(r["id"]),
            "category": r.get("category"),
            "answer_type": r.get("answer_type"),
            "question": r["question"],
            "gold": r["answer"],
            "prompt": f"{preamble}\n\n{r['question']}",
        })
    return items


def rebuild_prompts(subset_rows, with_tools=True):
    """Reattach prompts to a subset.json read back from a results directory.

    subset.json stores everything except the prompt, so a resume does not need the
    dataset -- but it does need the prompt text, which is a pure function of the
    question and the mode.
    """
    items = []
    for r in subset_rows:
        system = SYSTEM_MC if r.get("answer_type") == "multipleChoice" else SYSTEM_EXACT_ANSWER
        preamble = f"{system}\n\n{TOOL_PREAMBLE}" if with_tools else system
        it = dict(r)
        it["prompt"] = f"{preamble}\n\n{r['question']}"
        items.append(it)
    return items
