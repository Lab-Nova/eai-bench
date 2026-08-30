"""AA-LCR: fetching the dataset and assembling the 25-question subset.

Prompt assembly and document ordering follow the dataset card verbatim
(https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR).

AA's own 25-question subset is not published, so we take a deterministic, RNG-free 25
of the 100: sort by (document_category, document_set_id, question_id) and stride evenly
through that order, which spreads the sample across all seven document categories
instead of clustering on the 63-question Company set.
"""

import csv
import os
import urllib.request
import zipfile

N_SUBSET = 25
REPO = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main"
FILES = ("AA-LCR_Dataset.csv", "AA-LCR_extracted-text.zip")

PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


def ensure_data(data_dir):
    """Download the CSV and the document zip if they are not already present.

    Only the zip is ever fetched -- the extracted tree is derived from it, so a stale
    or partial extraction can always be repaired by deleting the directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(data_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{REPO}/{name}"
        print(f"fetching {url}", flush=True)
        tmp = dest + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as e:  # noqa: BLE001 - actionable message beats a traceback
            raise SystemExit(
                f"could not download {name}: {type(e).__name__}: {e}\n"
                f"AA-LCR is public (Apache-2.0); fetch it by hand into {data_dir} "
                f"or point --data-dir at an existing copy.")
        os.replace(tmp, dest)
    return data_dir


def ensure_extracted(data_dir):
    """Extract the document zip and return the directory holding the category folders."""
    docs_dir = os.path.join(data_dir, "extracted")
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        with zipfile.ZipFile(os.path.join(data_dir, "AA-LCR_extracted-text.zip")) as z:
            z.extractall(docs_dir)
    # The zip nests everything under a top folder; find the level holding the categories.
    for root, dirs, _ in os.walk(docs_dir):
        if any(os.path.isdir(os.path.join(root, d)) for d in dirs) and len(dirs) >= 5:
            return root
    return docs_dir


def load_rows(data_dir):
    with open(os.path.join(data_dir, "AA-LCR_Dataset.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["data_source_filenames"] = r["data_source_filenames"].split(";")
        r["input_tokens"] = int(r["input_tokens"])
    return rows


def pick_subset(rows, n):
    ordered = sorted(rows, key=lambda r: (r["document_category"], r["document_set_id"],
                                          int(r["question_id"])))
    total = len(ordered)
    if n > total:
        raise SystemExit(f"asked for {n} questions but the dataset has {total}")
    return [ordered[j * total // n] for j in range(n)]


def _key(name):
    """Encoding-insensitive filename key.

    The zip stores some names mojibake-encoded (U+2019 arrives as "ΓÇ–"),
    so exact matching against the CSV names fails. Comparing only the ASCII
    alphanumerics survives every such mangling.
    """
    return "".join(c for c in name if c.isascii() and c.isalnum()).lower()


_INDEX = {}


def find_doc(base, category, set_id, filename):
    direct = os.path.join(base, category, set_id, filename)
    if os.path.exists(direct):
        return direct
    d = os.path.join(base, category, set_id)
    if d not in _INDEX:
        idx = {}
        for n in os.listdir(d):
            idx.setdefault(_key(n), []).append(n)
        dupes = {k: v for k, v in idx.items() if len(v) > 1}
        if dupes:
            raise RuntimeError(f"ambiguous filename keys in {d}: {dupes}")
        _INDEX[d] = {k: v[0] for k, v in idx.items()}
    hit = _INDEX[d].get(_key(filename))
    if hit is None:
        raise FileNotFoundError(f"{category}/{set_id}/{filename}")
    return os.path.join(d, hit)


def build_items(data_dir, rows):
    """Turn CSV rows into items, reading and concatenating each question's documents."""
    base = ensure_extracted(data_dir)
    items = []
    for r in rows:
        docs = []
        for fn in r["data_source_filenames"]:
            with open(find_doc(base, r["document_category"], r["document_set_id"], fn),
                      encoding="utf-8") as f:
                docs.append(f.read())
        documents_text = "\n\n".join(
            f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
            for i, doc in enumerate(docs))
        items.append({
            "id": f"{r['document_set_id']}#{r['question_id']}",
            "document_category": r["document_category"],
            "document_set_id": r["document_set_id"],
            "question_id": r["question_id"],
            "question": r["question"],
            "gold": r["answer"],
            "declared_input_tokens": r["input_tokens"],
            "data_source_filenames": r["data_source_filenames"],
            "prompt": PROMPT_TEMPLATE.format(documents_text=documents_text,
                                             question=r["question"]),
        })
    return items


def load_subset(data_dir, n_subset=N_SUBSET, limit=None):
    ensure_data(data_dir)
    rows = load_rows(data_dir)
    subset = pick_subset(rows, n_subset)
    counts = {}
    for r in subset:
        counts[r["document_category"]] = counts.get(r["document_category"], 0) + 1
    print(f"selected {len(subset)} of {len(rows)}; categories: {counts}", flush=True)
    if limit:
        subset = subset[:limit]
    return build_items(data_dir, subset)


def rebuild_prompts(data_dir, subset_rows):
    """Reattach prompts to a subset.json read back from a results directory.

    subset.json stores the filenames but not the multi-megabyte document text, so a
    resume rebuilds the prompts from the local corpus rather than storing them twice.
    """
    ensure_data(data_dir)
    by_id = {f"{r['document_set_id']}#{r['question_id']}": r for r in load_rows(data_dir)}
    rows = []
    for s in subset_rows:
        r = by_id.get(str(s["id"]))
        if r is None:
            raise SystemExit(f"subset id {s['id']!r} is not in the AA-LCR CSV at {data_dir}")
        rows.append(r)
    return build_items(data_dir, rows)
