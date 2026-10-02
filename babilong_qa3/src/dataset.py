"""BABILong qa3: fetching / generating the splits and assembling the prompts.

qa3 ("three supporting facts") asks where an item was *before* a given location, so the
model has to chain three facts -- who picked the item up, where they went, where they
went next -- scattered through hundreds of thousands of tokens of PG19 text. Of the
BABILong tasks tried on GLM-5.3 it separated weight / KV-cache quantizations best: where
qa3 at 256k dropped about 15 points, qa1 moved 5-8 and qa5 / qa9 stayed within noise.

Lengths are GPT-2 tokens of the haystack, as in BABILong itself:

- 0k, 128k, 256k are the official splits, fetched from the HF dataset `RMT-team/babilong`
  pinned to a revision.
- 384k does not exist upstream, so it is generated here with BABILong's own recipe
  (create_tasks.py): seed 42 + 3*1000 + 384, bAbI en-10k qa3 train facts, PG19 test noise,
  `length - 300` tokens of message. The generator is deterministic, so every machine gets
  the same 100 samples; they are the ones the reference numbers were measured on.

The prompt is BABILong's default for qa3 -- instruction, in-context examples, the
answer-format post-prompt, then <context>...</context> and the question -- from
babilong/prompts.py, with the system prompt the BABILong leaderboard uses, except for one
change: the question. Upstream asks "Where was the milk before the hallway?", but in about
a quarter of the samples the item enters the hallway more than once, and the gold answer
is always the room it was in just before its *last* entry. Nothing in the prompt says so,
and GLM-5.3 answers for the first entry often enough to cost 1-4 points at 0k. So every
question is rewritten to say it (`clarify`), and the examples follow suit, with a third
one in which the item revisits the asked room. Scores are therefore not comparable with
upstream BABILong numbers.
"""

import json
import os
import re
import urllib.request
import zipfile

TASK = "qa3"
LENGTHS = ("0k", "128k", "256k", "384k")
N_PER_LENGTH = 100

HF_REVISION = "ee0d588794c7ac098062ee0d247c733d62e94fe2"
HF_SPLIT_URL = ("https://huggingface.co/datasets/RMT-team/babilong/resolve/"
                f"{HF_REVISION}/data/{TASK}/{{length}}.json")
HF_LENGTHS = {"0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"}
BABI_ZIP_URL = ("https://raw.githubusercontent.com/booydar/babilong/"
                "7a6efee29f5cac03c3c410e6799c80fd2ffe3610/data/tasks_1-20_v1-2.zip")
BABI_TRAIN_FILE = "tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt"
PG19 = ("emozilla/pg19-test", "c5e39bf32e33f9111323aa68d7d9000d22722035")
SEED = 42

SYSTEM_PROMPT = "You are a helpful AI assistant."

# babilong/prompts.py: DEFAULT_TEMPLATE and DEFAULT_PROMPTS["qa3"], verbatim except that the
# example questions use the clarified wording and a third example shows a revisit.
TEMPLATE = ("{instruction}\n\n{examples}\n\n{post_prompt}\n\n"
            "<context>\n{context}\n</context>\n\nQuestion: {question}")
INSTRUCTION = (
    'I give you context with the facts about locations and actions of different persons '
    'hidden in some random text and a question. '
    'You need to answer the question based only on the information from the facts.\n'
    'If a person got an item in the first location and travelled to the second location '
    'the item is also in the second location. '
    'If a person dropped an item in the first location and moved to the second location '
    'the item remains in the first location.')
EXAMPLES = (
    '<example>\n'
    'John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. '
    'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
    'Where was the apple just before it was last carried into the kitchen?\n'
    'Answer: Before the kitchen the apple was in the bathroom.\n'
    '</example>\n'
    '<example>\n'
    'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
    'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
    'Where was the football just before it was last carried into the bedroom?\n'
    'Answer: Before the bedroom the football was in the garden.\n'
    '</example>\n'
    '<example>\n'
    'Mary journeyed to the kitchen. Mary took the milk. Mary went to the office. '
    'Mary moved to the garden. Mary travelled to the hallway. Mary went back to the garden. '
    'Where was the milk just before it was last carried into the garden?\n'
    'Answer: Before the garden the milk was in the hallway.\n'
    '</example>')
POST_PROMPT = (
    'Always return your answer in the following format: '
    'Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.')


def parse_length(name):
    name = name.lower()
    if name.endswith("m"):
        return int(float(name[:-1]) * 1_000_000)
    return int(float(name.rstrip("k")) * 1000)


UPSTREAM_QUESTION = re.compile(r"Where was the (\w+) before the (\w+)\?\s*")


def clarify(question):
    """Upstream's "Where was the X before the Y?" -> the unambiguous last-entry form."""
    m = UPSTREAM_QUESTION.fullmatch(question)
    if not m:
        raise ValueError(f"unexpected qa3 question: {question!r}")
    return f"Where was the {m[1]} just before it was last carried into the {m[2]}?"


def format_prompt(context, question):
    return TEMPLATE.format(instruction=INSTRUCTION, examples=EXAMPLES, post_prompt=POST_PROMPT,
                           context=context.strip(), question=question).strip()


def _fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"fetching {url}", flush=True)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:  # noqa: BLE001 - actionable message beats a traceback
        raise SystemExit(f"could not download {url}: {type(e).__name__}: {e}\n"
                         f"fetch it by hand into {dest} or point --data-dir at a copy.")
    os.replace(tmp, dest)
    return dest


class _GPT2:
    """The transformers-tokenizer calls babi_noise makes, on the `tokenizers` library."""

    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_pretrained("gpt2")

    def encode(self, text, add_special_tokens=False):
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts, add_special_tokens)}
        return {"input_ids": [e.ids for e in
                              self.tok.encode_batch(texts, add_special_tokens=add_special_tokens)]}

    def decode(self, ids):
        return self.tok.decode(ids)


def generate_split(data_dir, len_name, n):
    """BABILong's create_tasks.py recipe at an arbitrary length; the first n samples.

    The sample order is a seeded shuffle of the whole task, so any n is a prefix of any
    larger n: asking for fewer samples never changes the ones you get.
    """
    import datasets
    import numpy as np
    from babi_noise import NoiseInjectionDataset, SentenceSampler, TaskDataset

    babi_dir = os.path.join(data_dir, "babi")
    zip_path = _fetch(BABI_ZIP_URL, os.path.join(babi_dir, "tasks_1-20_v1-2.zip"))
    task_path = os.path.join(babi_dir, BABI_TRAIN_FILE)
    if not os.path.exists(task_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extract(BABI_TRAIN_FILE, babi_dir)

    length = parse_length(len_name)
    seed = SEED + int(TASK[2:]) * 1000 + length // 1000
    np.random.seed(seed)
    tokenizer = _GPT2()
    noise = datasets.load_dataset(PG19[0], split="test", revision=PG19[1])
    message_length = max(length - 300, 0)  # leave room for the prompt, as create_tasks.py
    task_dataset = (TaskDataset(task_path, max_n_facts=message_length // 8)
                    if message_length > 0 else TaskDataset(task_path))
    sampler = SentenceSampler(noise, tokenizer=tokenizer, shuffle=True, random_seed=seed)
    ds = NoiseInjectionDataset(task_dataset=task_dataset, noise_sampler=sampler,
                               tokenizer=tokenizer, sample_size=message_length, random_seed=seed)
    inds = list(range(len(ds)))
    np.random.shuffle(inds)
    out = []
    for k, i in enumerate(inds[:n]):
        s = ds[i]
        out.append({"input": tokenizer.decode(s["input_tokens"]).strip(),
                    "question": s["question"], "target": tokenizer.decode(s["target_tokens"])})
        print(f"  {TASK} {len_name}: generated {k + 1}/{n}", flush=True)
    return out


def load_split(data_dir, len_name, n):
    """The first n samples of one length, fetched or generated into data_dir/qa3/."""
    path = os.path.join(data_dir, TASK, f"{len_name}.json")
    if len_name in HF_LENGTHS:
        _fetch(HF_SPLIT_URL.format(length=len_name), path)
    elif not os.path.exists(path) or len(_read(path)) < n:
        print(f"{TASK} {len_name} is not an official split; generating {n} samples "
              f"(deterministic, seed {SEED} + 3000 + {parse_length(len_name) // 1000})",
              flush=True)
        samples = generate_split(data_dir, len_name, n)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(samples, f)
        os.replace(tmp, path)
    samples = _read(path)
    if len(samples) < n:
        raise SystemExit(f"{path} holds {len(samples)} samples, fewer than the {n} asked for")
    return samples[:n]


def _read(path):
    with open(path) as f:
        return json.load(f)


def build_items(data_dir, keys):
    """Items for (length, idx) keys, in the given order."""
    need = {}
    for ln, idx in keys:
        need[ln] = max(need.get(ln, 0), idx + 1)
    splits = {ln: load_split(data_dir, ln, n) for ln, n in need.items()}
    items = []
    for ln, idx in keys:
        s = splits[ln][idx]
        q = clarify(s["question"])
        items.append({"id": f"{TASK}-{ln}-{idx}", "length": ln, "idx": idx,
                      "question": q, "target": s["target"],
                      "prompt": format_prompt(s["input"], q)})
    return items


def load_subset(data_dir, lengths=LENGTHS, n_per_length=N_PER_LENGTH, limit=None):
    keys = [(ln, i) for ln in lengths for i in range(n_per_length)]
    if limit:
        keys = keys[:limit]
    return build_items(data_dir, keys)


def rebuild_prompts(data_dir, subset_rows):
    """Reattach prompts to a subset.json read back from a results directory."""
    return build_items(data_dir, [(s["length"], int(s["idx"])) for s in subset_rows])
