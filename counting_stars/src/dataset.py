"""Counting-Stars: fetching the inputs and building the long-context prompts.

The correction test ("reasoning", Me-Rea.) of Counting-Stars
(https://github.com/nick7nlp/Counting-Stars), in English and Chinese. Each prompt holds
32 evenly spaced needles of the form

    EN: "The little penguin counted 16 ★, but found that a mistake had been made, so the
         counting was done again, and this time 15 ★ was counted correctly."
    ZH: "小企鹅数了16颗★，但发现数错了，于是又数了一遍，这次数对了，是15颗★"

followed by the upstream question, which asks for the list of *correct* counts as JSON.

Two departures from the 128k files shipped upstream, both needed to go past 128k:

- Lengths are GLM-5.3 tokens of the user message, cut with the model's own tokenizer,
  not words / characters.
- English filler is PG19 test books: the Paul Graham essays upstream uses run out at
  roughly 140k tokens. Chinese filler is upstream's own (The Story of the Stone + Journey
  to the West), cleaned as in its gen_test_data.ipynb.

Generation is deterministic: the same inputs give byte-identical prompts on any machine.
Every input is pinned to a revision and fetched into `--data-dir` on first use; built
prompts are cached there as prompts/reasoning_{lang}_{length}.json.
"""

import hashlib
import json
import os
import re
import urllib.request

M = 32  # needles per prompt
TASK = "reasoning"
LANGS = ("EN", "ZH")
LENGTHS = tuple(f"{64 * i}k" for i in range(1, 16))  # 64k .. 960k
# Upstream has one star set per needle count, so each (lang, length) has exactly one
# prompt. A single 32-needle prompt is a noisy measurement -- a single prompt's score moved by
# up to 0.3 between endpoints whose overall means agreed -- so the standard run samples every
# prompt this many times, independently, at the client's sampling temperature.
N_SAMPLES = 5

UPSTREAM = ("https://raw.githubusercontent.com/nick7nlp/Counting-Stars/"
            "32e16f2c2d2be9f38d710e782236315e0821a038/context_data")
CONTEXT_FILES = ("a_stars.txt", "r_stars.txt",
                 "The_Story_of_the_Stone.txt", "Journey_to_the_West.txt")
# Lengths are counted in this tokenizer. The sha256 pins the exact file: it is the one
# shipped with GLM-5.3 (and GLM-5.3-BF16), so a revision bump upstream cannot silently
# move every needle.
TOKENIZER_URL = ("https://huggingface.co/zai-org/GLM-5.3/resolve/"
                 "aca966e4e02791568aa6a4ced368624b3d897f42/tokenizer.json")
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
PG19 = ("emozilla/pg19-test", "c5e39bf32e33f9111323aa68d7d9000d22722035")

QUESTIONS = {
    "ZH": "\n\n\n\n在这个月光皎洁、云雾缭绕的夜晚，小企鹅正望向天空，全神贯注地数★。请帮助小企鹅收集所数★的正确颗数，按照如下格式：{\"小企鹅\":[x,x,x,...]}，不要求和，[x,x,x,...]中数字为小企鹅正确数★的颗数，仅以JSON格式输出结果，不需要输出任何解释。",
    "EN": "\n\n\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the correct number of ★, for example: {\"little_penguin\": [x, x, x,...]}. The summation is not required, and the numbers in [x, x, x,...] represent the correctly counted number of ★ by the little penguin. Only output the results in JSON format without any explanation.",
}


def parse_length(name):
    name = name.lower()
    if name.endswith("m"):
        return int(float(name[:-1]) * 1_000_000)
    return int(float(name.rstrip("k")) * 1000)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url, dest, sha256=None):
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"fetching {url}", flush=True)
        tmp = dest + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as e:  # noqa: BLE001 - actionable message beats a traceback
            raise SystemExit(f"could not download {url}: {type(e).__name__}: {e}\n"
                             f"fetch it by hand into {dest} or point --data-dir at a copy.")
        os.replace(tmp, dest)
    if sha256 and _sha256(dest) != sha256:
        raise SystemExit(f"{dest} does not match the pinned sha256 {sha256}; delete it to refetch")
    return dest


def ensure_inputs(data_dir):
    for name in CONTEXT_FILES:
        _fetch(f"{UPSTREAM}/{name}", os.path.join(data_dir, "context_data", name))
    _fetch(TOKENIZER_URL, os.path.join(data_dir, "tokenizer.json"), TOKENIZER_SHA256)


def load_tokenizer(data_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))


def load_stars(data_dir):
    """Upstream stores one Python dict literal per file, keyed by needle count."""
    import ast
    out = []
    for name in ("a_stars.txt", "r_stars.txt"):
        with open(os.path.join(data_dir, "context_data", name), encoding="utf-8") as f:
            out.append(ast.literal_eval(f.readline())[str(M)])
    return tuple(out)


def needle(lang, a, r):
    if lang == "ZH":
        return f"\n小企鹅数了{r}颗★，但发现数错了，于是又数了一遍，这次数对了，是{a}颗★\n"
    return (f"\nThe little penguin counted {r} ★, but found that a mistake had been made, "
            f"so the counting was done again, and this time {a} ★ was counted correctly.\n")


def filler_text(data_dir, lang, min_tokens):
    """Haystack text with at least min_tokens tokens (the same prefix for every length)."""
    if lang == "ZH":
        text = ""
        for name in ("The_Story_of_the_Stone.txt", "Journey_to_the_West.txt"):
            with open(os.path.join(data_dir, "context_data", name), encoding="utf-8") as f:
                for line in f:
                    text += line.strip().replace("------------", " ").replace(" ", "")
        text = re.sub("[{}]".format(re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")), "", text)
        return re.sub("[a-zA-Z]", "", text)
    import datasets
    books = datasets.load_dataset(PG19[0], split="test", revision=PG19[1])
    parts, n = [], 0
    for book in books:
        t = re.sub(r"\s+", " ", book["text"]).strip()
        parts.append(t)
        n += len(t) // 4  # rough, only to know when to stop reading books
        if n > min_tokens * 1.3:
            break
    return " ".join(parts)


def build_prompt(data_dir, lang, length, tokenizer, filler, filler_offsets):
    a_stars, r_stars = load_stars(data_dir)
    needles = [needle(lang, a_stars[i], r_stars[i]) for i in range(M)]
    question = QUESTIONS[lang]
    n_fixed = (len(tokenizer.encode(question).ids)
               + sum(len(tokenizer.encode(s).ids) for s in needles))
    n_filler = length - n_fixed
    if n_filler > len(filler_offsets):
        raise ValueError(f"{lang} filler has {len(filler_offsets)} tokens, need {n_filler}")

    # Needle i goes after filler token (i+1)*n_filler/M, so the last one sits right before
    # the question, as in gen_test_data.ipynb. English cut points move to the next space so
    # needles fall between words.
    out, prev = [], 0
    for i in range(M):
        cut = filler_offsets[round((i + 1) * n_filler / M) - 1][1]
        if lang == "EN":
            sp = filler.find(" ", cut)
            cut = sp if sp != -1 else cut
        out.append(filler[prev:cut])
        out.append(needles[i])
        prev = cut + (1 if lang == "EN" and filler[cut:cut + 1] == " " else 0)
    prompt = "".join(out) + question
    return {"lang": lang, "task": TASK, "length": length,
            "prompt_tokens": len(tokenizer.encode(prompt).ids),
            "reference_counting_results": a_stars, "wrong_counting_results": r_stars,
            "question": prompt}


def get_prompt(data_dir, lang, len_name, max_len, cache):
    """One prompt, from the cache under data_dir/prompts or built and cached."""
    path = os.path.join(data_dir, "prompts", f"{TASK}_{lang}_{len_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if "tokenizer" not in cache:
        ensure_inputs(data_dir)
        cache["tokenizer"] = load_tokenizer(data_dir)
    tok = cache["tokenizer"]
    if lang not in cache:
        text = filler_text(data_dir, lang, max_len)
        cache[lang] = (text, tok.encode(text).offsets)
        print(f"{lang} filler: {len(text)} chars, {len(cache[lang][1])} tokens", flush=True)
    p = build_prompt(data_dir, lang, parse_length(len_name), tok, *cache[lang])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"wrote {path} ({p['prompt_tokens']} tokens)", flush=True)
    return p


def build_items(data_dir, keys):
    """Items for (lang, length, sample) keys, in the given order."""
    ensure_inputs(data_dir)
    max_len = max(parse_length(ln) for _, ln, _ in keys)
    cache, prompts, items = {}, {}, []
    for lang, len_name, k in keys:
        if (lang, len_name) not in prompts:
            prompts[(lang, len_name)] = get_prompt(data_dir, lang, len_name, max_len, cache)
        p = prompts[(lang, len_name)]
        items.append({
            "id": f"{lang}-{len_name}-s{k}",
            "lang": lang,
            "length": len_name,
            "sample": k,
            "prompt_tokens": p["prompt_tokens"],
            "reference": p["reference_counting_results"],
            "wrong": p["wrong_counting_results"],
            "prompt": p["question"],
        })
    return items


def load_subset(data_dir, langs=LANGS, lengths=LENGTHS, n_samples=N_SAMPLES, limit=None):
    keys = [(lang, ln, k) for lang in langs for ln in lengths for k in range(n_samples)]
    if limit:
        keys = keys[:limit]
    return build_items(data_dir, keys)


def rebuild_prompts(data_dir, subset_rows):
    """Reattach prompts to a subset.json read back from a results directory."""
    return build_items(data_dir, [(s["lang"], s["length"], int(s["sample"])) for s in subset_rows])
