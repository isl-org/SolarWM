"""Generate RULER synthetic tasks fitted to Qwen3-Omni chat-templated token lengths.

The per-task generators are ports of NVIDIA RULER's
third_party/RULER/scripts/data/synthetic/{niah,variable_tracking,common_words_extraction,
freq_words_extraction,qa}.py (task args from scripts/synthetic.yaml, templates and
tokens_to_generate from scripts/data/synthetic/constants.py). The prompt text, needle
formats, haystacks, answer prefixes and gold answers follow RULER exactly. What differs is
the length fitting:

  * RULER fits `len(tokenizer(input)) + tokens_to_generate <= max_seq_length` with one
    binary search for the whole task. Here every sample is fitted on its own so that
    len(apply_chat_template([{"role": "user", "content": input}], add_generation_prompt=True))
    is in [floor(0.97 * L), L], using the Qwen3-Omni tokenizer. L already leaves room
    for generation, so tokens_to_generate is not subtracted.
  * RNG is seeded per (task, length, index, attempt), so each sample can be reproduced on
    its own. If a sample cannot land in the window (for example, qa document
    granularity), the distractor draw is resampled with attempt + 1.

Usage:
  python generate.py [--lengths 2048 8192 ...] [--tasks niah_single_1 ...] [--num_samples 20]
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import string
import sys
import time
import uuid

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RULER = os.path.join(HERE, "..", "third_party", "RULER")
SYN_DIR = os.path.join(RULER, "scripts", "data", "synthetic")
JSON_DIR = os.path.join(SYN_DIR, "json")
TOKENIZER_ID = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"

ALL_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
             "niah_multikey_2", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
             "qa_1", "qa_2"]
ALL_LENGTHS = [2048, 8192, 16384, 32768, 63488]
LOW_FRAC = 0.97

# --------------------------------------------------------------------------- RULER config
def load_ruler_constants():
    # constants.py is a plain dict literal; exec it in isolation instead of importing.
    ns = {}
    with open(os.path.join(SYN_DIR, "constants.py")) as f:
        exec(compile(f.read(), "ruler_constants", "exec"), ns)
    return ns["TASKS"]


TASKS_BASE = load_ruler_constants()
with open(os.path.join(RULER, "scripts", "synthetic.yaml")) as f:
    TASKS_CFG = yaml.safe_load(f)


def task_config(name):
    cfg = dict(TASKS_CFG[name])
    cfg.update(TASKS_BASE[cfg["task"]])
    # prepare.py with model_template_type=base: template = task_template + answer_prefix
    cfg["full_template"] = cfg["template"] + cfg.get("answer_prefix", "")
    return cfg


def split_answer_prefix(text, task_kind):
    """Same as RULER: locate the answer prefix by its first 10 chars (rfind)."""
    ap = TASKS_BASE[task_kind]["answer_prefix"]
    i = text.rfind(ap[:10])
    return text[:i], text[i:]


# --------------------------------------------------------------------------- shared data
_cache = {}


def essay_words():
    if "essay" not in _cache:
        essay = json.load(open(os.path.join(JSON_DIR, "PaulGrahamEssays.json")))["text"]
        _cache["essay"] = re.sub(r"\s+", " ", essay).split(" ")
    return _cache["essay"]


def wonder(name):
    import wonderwords
    return wonderwords.random_word._get_words_from_text_file(name)


def niah_words():
    if "niah_words" not in _cache:
        nouns, adjs = wonder("nounlist.txt"), wonder("adjectivelist.txt")
        _cache["niah_words"] = sorted(set(f"{a}-{n}" for a in adjs for n in nouns))
    return _cache["niah_words"]


NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))
BASE_SEED = 42


def sent_tokenize(text):
    from nltk.tokenize import sent_tokenize as st
    return st(text)


# --------------------------------------------------------------------------- niah
def niah_gen(cfg, h, rng):
    a = cfg["args"]
    num_k = max(a["num_needle_k"], a["num_needle_q"])
    num_v, num_q = a["num_needle_v"], a["num_needle_q"]
    words = niah_words()

    def rnd(kind):
        if kind == "numbers":
            return str(rng.randint(10 ** 6, 10 ** 7 - 1))
        if kind == "words":
            return rng.choice(words)
        if kind == "uuids":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        raise NotImplementedError(kind)

    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    keys, values, needles = [], [], []
    for _ in range(num_k):
        keys.append(rnd(a["type_needle_k"]))
        value = []
        for _ in range(num_v):
            value.append(rnd(a["type_needle_v"]))
            needles.append(needle.format(type_needle_v=a["type_needle_v"], key=keys[-1], value=value[-1]))
        values.append(value)
    random.Random(BASE_SEED).shuffle(needles)

    if a["type_haystack"] == "essay":
        hay = essay_words()
        if h <= len(hay):
            text = " ".join(hay[:h])
        else:
            text = " ".join((hay * ((h + len(hay) - 1) // len(hay)))[:h])
        sents = sent_tokenize(text.strip())
        pos = [0] + sorted(int(len(sents) * (d / 100)) for d in rng.sample(DEPTHS, len(needles))) + [len(sents)]
        parts = []
        for i in range(1, len(pos)):
            parts.append(" ".join(sents[pos[i - 1]:pos[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)
    else:
        if a["type_haystack"] == "noise":
            sentences = [NOISE] * h
        elif a["type_haystack"] == "needle":
            sentences = [needle.format(type_needle_v=a["type_needle_v"], key=rnd(a["type_needle_k"]),
                                       value=rnd(a["type_needle_v"])) for _ in range(h)]
        else:
            raise NotImplementedError(a["type_haystack"])
        idx = sorted(rng.sample(range(h), len(needles)), reverse=True)
        for i, el in zip(idx, needles):
            sentences.insert(i, el)
        context = "\n".join(sentences)

    indices = rng.sample(range(num_k), num_q)
    queries = [keys[i] for i in indices]
    answers = [x for i in indices for x in values[i]]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

    template = cfg["full_template"]
    tnv = a["type_needle_v"]
    if num_q * num_v == 1:
        template = (template.replace("Some", "A").replace("are all", "is").replace("are", "is")
                    .replace("answers", "answer"))
        tnv = tnv[:-1]
    full = template.format(type_needle_v=tnv, context=context, query=query)
    inp, ap = split_answer_prefix(full, "niah")
    return inp, ap, answers


# --------------------------------------------------------------------------- variable tracking
def vt_chains(rng, nprng, num_chains, num_hops, is_icl):
    k = 3 if is_icl else 5
    if is_icl:
        num_hops = min(10, num_hops)
    vars_all = ["".join(rng.choices(string.ascii_uppercase, k=k)).upper() for _ in range((num_hops + 1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)).upper())
    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        tv = vars_all[i:i + num_hops + 1]
        vars_ret.append(tv)
        chain = [f"VAR {tv[0]} = 12345"] if is_icl else [f"VAR {tv[0]} = {str(nprng.randint(10000, 99999))}"]
        for j in range(num_hops):
            chain.append(f"VAR {tv[j + 1]} = VAR {tv[j]} ")
        chains_ret.append(chain)
    return vars_ret, chains_ret


def vt_body(cfg, h, rng, nprng, is_icl):
    a = cfg["args"]
    assert a["type_haystack"] == "noise"
    vars_, chains = vt_chains(rng, nprng, a["num_chains"], a["num_hops"], is_icl)
    value = chains[0][0].split("=")[-1].strip()
    sentences = [NOISE] * h
    for chain in chains:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for p, j in zip(positions, range(len(chain))):
            sentences.insert(p + j, chain[j])
    context = "\n".join(sentences).replace(". \n", ".\n")
    text = cfg["full_template"].format(context=context, query=value, num_v=a["num_hops"] + 1)
    return text, vars_[0]


def vt_icl_example(cfg, count_raw, seed):
    """RULER: one ICL example fitted to 500 tokens (raw, no chat template, no generation budget)."""
    rng, nprng = random.Random(seed), np.random.RandomState(seed % 2 ** 32)
    best = None
    for h in range(5, 400, 5):
        r2, n2 = random.Random(seed), np.random.RandomState(seed % 2 ** 32)
        text, ans = vt_body(cfg, h, r2, n2, is_icl=True)
        if count_raw(text) <= 500:
            best = (text, ans)
        else:
            break
    text, ans = best
    return text + " " + " ".join(ans) + "\n"


def vt_randomize_icl(icl, num_hops, rng, nprng):
    tgt = icl.strip().split()[-num_hops - 1:]
    for item in tgt:
        icl = icl.replace(item, "".join(rng.choices(string.ascii_uppercase, k=len(item))).upper())
    return icl.replace("12345", str(nprng.randint(10000, 99999)))


def vt_gen(cfg, h, rng, nprng, icl):
    text, ans = vt_body(cfg, h, rng, nprng, is_icl=False)
    cutoff = text.index(TASKS_BASE["variable_tracking"]["template"][:20])
    text = text[:cutoff] + vt_randomize_icl(icl, cfg["args"]["num_hops"], rng, nprng) + "\n" + text[cutoff:]
    inp, ap = split_answer_prefix(text, "variable_tracking")
    return inp, ap, ans


# --------------------------------------------------------------------------- cwe
def cwe_words():
    if "cwe_words" not in _cache:
        w = sorted(set(wonder("nounlist.txt") + wonder("adjectivelist.txt") + wonder("verblist.txt")))
        random.Random(BASE_SEED).shuffle(w)
        _cache["cwe_words"] = w
    return _cache["cwe_words"]


def cwe_words_large():
    """RULER's english_words.json (git-lfs object), used when a sample needs more distinct words than the
    wonderwords lists hold (very long contexts)."""
    if "cwe_words_large" not in _cache:
        d = json.load(open(os.path.join(JSON_DIR, "english_words.json")))
        if not isinstance(d, dict):
            raise RuntimeError("english_words.json is a git-lfs pointer; fetch the object first")
        w = sorted(set(d.values()))
        random.Random(BASE_SEED).shuffle(w)
        _cache["cwe_words_large"] = w
    return _cache["cwe_words_large"]


def cwe_gen(cfg, h, rng, L):
    a = cfg["args"]
    words = cwe_words()

    def example(num_words, rep_c, rep_u, num_c):
        nonlocal words
        if num_words > len(words):
            words = cwe_words_large()
        if num_words > len(words):
            raise RuntimeError(f"cwe needs {num_words} distinct words")
        full = rng.sample(words, num_words)
        common, uncommon = full[:num_c], full[num_c:]
        wl = common * int(rep_c) + uncommon * int(rep_u)
        random.Random(BASE_SEED).shuffle(wl)
        return " ".join(f"{i + 1}. {w}" for i, w in enumerate(wl)), common

    tmpl = cfg["full_template"]
    num_fewshot = 1
    shots = []
    # RULER switches on max_seq_length < 4096 (prompt + generation); we pass the bucket L.
    if L < 4096:
        for _ in range(num_fewshot):
            shots.append(example(20, 3, 1, a["num_cw"]))
        context, answer = example(h, 6, 1, a["num_cw"])
    else:
        for _ in range(num_fewshot):
            shots.append(example(40, 10, 3, a["num_cw"]))
        context, answer = example(h, a["freq_cw"], a["freq_ucw"], a["num_cw"])
    shots = [tmpl.format(num_cw=a["num_cw"], context=c, query="") + " " +
             " ".join(f"{i + 1}. {w}" for i, w in enumerate(ans)) for c, ans in shots]
    text = "\n".join(shots) + "\n" + tmpl.format(num_cw=a["num_cw"], context=context, query="")
    inp, ap = split_answer_prefix(text, "common_words_extraction")
    return inp, ap, answer


# --------------------------------------------------------------------------- fwe
def fwe_gen(cfg, h, rng, L):
    from scipy.special import zeta
    alpha = float(cfg["args"]["alpha"])
    coded_wordlen = 6
    # RULER: vocab_size = (max_seq_length - tokens_to_generate) // 50
    vocab_size = (L - cfg["tokens_to_generate"]) // 50
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=coded_wordlen)) for _ in range(vocab_size)]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(rng.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(set(vocab))
    random.Random(BASE_SEED).shuffle(vocab)
    vocab[0] = "..."
    k = np.arange(1, len(vocab) + 1)
    cnt = h * (k ** -alpha) / zeta(alpha)
    sampled = [x for w, z in zip(vocab, cnt.astype(int)) for x in [w] * z]
    random.Random(BASE_SEED).shuffle(sampled)
    text = cfg["full_template"].format(context=" ".join(sampled), query="")
    inp, ap = split_answer_prefix(text, "freq_words_extraction")
    return inp, ap, vocab[1:4]


# --------------------------------------------------------------------------- qa
def read_squad(file):
    data = json.load(open(file))
    docs = sorted(set(p["context"] for d in data["data"] for p in d["paragraphs"]))
    dd = {c: i for i, c in enumerate(docs)}
    qas = []
    for d in data["data"]:
        more = [dd[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for q in p["qas"]:
                if not q["is_impossible"]:
                    qas.append({"query": q["question"], "outputs": [x["text"] for x in q["answers"]],
                                "context": [dd[p["context"]]],
                                "more_context": [i for i in more if i != dd[p["context"]]]})
    return qas, docs


def read_hotpotqa(file):
    data = json.load(open(file))
    docs = sorted(set(f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]))
    dd = {c: i for i, c in enumerate(docs)}
    qas = [{"query": d["question"], "outputs": [d["answer"]],
            "context": [dd[f"{t}\n{''.join(p)}"] for t, p in d["context"]]} for d in data]
    return qas, docs


def qa_data(dataset):
    key = "qa_" + dataset
    if key not in _cache:
        _cache.clear()  # only keep one QA corpus in RAM at a time
        if dataset == "squad":
            _cache[key] = read_squad(os.path.join(JSON_DIR, "squad.json"))
        else:
            _cache[key] = read_hotpotqa(os.path.join(JSON_DIR, "hotpotqa.json"))
    return _cache[key]


def qa_gen(cfg, h, rng, qidx):
    qas, docs = qa_data(cfg["args"]["dataset"])
    q = qas[qidx]  # RULER: sample i uses question i + pre_samples (0); see qa_skip in generate_task
    cur, more = q["context"], q.get("more_context", [])
    h = max(h, len(cur))
    if h < len(docs):
        if h - len(cur) > len(more):
            excl = set(cur + more)
            # RULER builds this list each call; equivalent membership test with a set
            addition = [i for i in range(len(docs)) if i not in excl]
            all_docs = cur + more + rng.sample(addition, max(0, h - len(cur) - len(more)))
        else:
            all_docs = cur + rng.sample(more, h - len(cur))
        all_docs = [docs[i] for i in all_docs]
    else:
        all_docs = (docs * ((h + len(docs) - 1) // len(docs)))[:h]
    random.Random(BASE_SEED).shuffle(all_docs)
    context = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(all_docs))
    text = cfg["full_template"].format(context=context, query=q["query"])
    inp, ap = split_answer_prefix(text, "qa")
    return inp, ap, q["outputs"]


# --------------------------------------------------------------------------- fitting
class Counter:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        probe = self.tok.apply_chat_template([{"role": "user", "content": "XYZ"}], tokenize=False,
                                             add_generation_prompt=True)
        self.prefix, self.suffix = probe.split("XYZ")

    def raw(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def templated(self, user_text):
        return self.raw(self.prefix + user_text + self.suffix)

    def templated_official(self, user_text):
        enc = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=True,
                                           add_generation_prompt=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        return len(ids)


def seed_for(task, L, index, attempt):
    return int(hashlib.md5(f"{BASE_SEED}|{task}|{L}|{index}|{attempt}".encode()).hexdigest()[:12], 16)


def fit_sample(make, count, L, h_init, h_min=1, accept_frac=0.985):
    """Find h with floor(0.97L) <= count(make(h)) <= L. make(h) must be deterministic."""
    lo_ok = math.floor(LOW_FRAC * L)
    memo = {}

    def ev(h):
        if h not in memo:
            out = make(h)
            memo[h] = (count(out[0]), out)
        return memo[h][0]

    good, bad = None, None  # good: largest h with n<=L ; bad: smallest h with n>L
    h = max(h_min, int(h_init))
    for _ in range(6):  # secant-style jumps toward ~0.99 L
        n = ev(h)
        if n <= L:
            good = h if good is None or h > good else good
            if n >= accept_frac * L:
                return h, memo[h]
        else:
            bad = h if bad is None or h < bad else bad
        nh = max(h_min, int(h * (0.99 * L) / max(n, 1)))
        if nh == h:
            nh = h + 1 if n <= L else h - 1
        if good is not None and bad is not None:
            break
        h = max(h_min, nh)
    while good is None:
        h = max(h_min, int((bad or h) * 0.9))
        if ev(h) <= L:
            good = h
        else:
            bad = h
            if h == h_min:
                return None, None
    while bad is None:
        h = int(good * 1.1) + 1
        if ev(h) > L:
            bad = h
        else:
            good = h
    while bad - good > 1:
        if memo[good][0] >= accept_frac * L:
            break
        mid = (good + bad) // 2
        if ev(mid) <= L:
            good = mid
        else:
            bad = mid
    n = memo[good][0]
    if n < lo_ok:
        return None, (n, good)
    return good, memo[good]


def generate_task(task, L, num_samples, counter):
    cfg = task_config(task)
    kind = cfg["task"]
    rows = []
    icl = None
    if kind == "variable_tracking":
        icl = vt_icl_example(cfg, counter.raw, seed_for(task, L, -1, 0))
    if kind == "freq_words_extraction":
        h_init = L // 6
    elif kind == "qa":
        h_init = max(10, L // 150)
    elif kind == "common_words_extraction":
        h_init = max(20, L // 20 if L < 4096 else L // 25)
    elif kind == "niah" and cfg["args"]["type_haystack"] == "essay":
        h_init = int(L * 0.7)
    else:
        h_init = max(5, L // 25)
    h_min = {"qa": 1, "common_words_extraction": 11}.get(kind, 5)

    # RULER uses question i for sample i. If a question's own gold docs already exceed L (HotpotQA
    # ships 10 paragraphs per question; this happens at 2048), RULER's retry loop would never end.
    # We skip such questions and move on to the next unused question index.
    qa_skip = 0
    for index in range(num_samples):
        attempt = 0
        while attempt < 50:
            seed = seed_for(task, L, index, attempt)

            def make(h, seed=seed):
                rng = random.Random(seed)
                nprng = np.random.RandomState(seed % 2 ** 32)
                if kind == "niah":
                    return niah_gen(cfg, h, rng)
                if kind == "variable_tracking":
                    return vt_gen(cfg, h, rng, nprng, icl)
                if kind == "common_words_extraction":
                    return cwe_gen(cfg, h, rng, L)
                if kind == "freq_words_extraction":
                    return fwe_gen(cfg, h, rng, L)
                if kind == "qa":
                    return qa_gen(cfg, h, rng, index + qa_skip)
                raise NotImplementedError(kind)

            h, res = fit_sample(make, counter.templated, L, h_init, h_min)
            if h is not None:
                break
            if res is None and kind == "qa":
                print(f"  {task} L={L} idx={index}: question {index + qa_skip} gold docs alone exceed L, "
                      f"skipping to next question", flush=True)
                qa_skip += 1
                attempt = 0
                continue
            print(f"  {task} L={L} idx={index} attempt {attempt}: best {res} below window, resampling",
                  flush=True)
            attempt += 1
        else:
            raise RuntimeError(f"could not fit {task} L={L} idx={index}")
        h_init = h
        _, (inp, ap, outputs) = res
        n_off = counter.templated_official(inp)
        assert math.floor(LOW_FRAC * L) <= n_off <= L, (task, L, index, n_off)
        row = {"task": task, "index": index, "length_bucket": L, "input": inp,
               "n_tokens_templated": n_off, "outputs": list(outputs),
               "max_new_tokens": cfg["tokens_to_generate"], "answer_prefix": ap,
               "haystack_units": h, "seed": seed}
        if kind == "qa":
            row["source_qa_index"] = index + qa_skip
        rows.append(row)
    return rows


def main():
    global TOKENIZER_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=ALL_LENGTHS)
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID,
                    help="HF id or local dir whose tokenizer + chat template define the length fit "
                         "(default: Qwen3-Omni; Qwen3-VL: the quantised checkpoint dir)")
    args = ap.parse_args()
    TOKENIZER_ID = args.tokenizer

    parts = os.path.join(args.out_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    counter = Counter()
    # Group by task so the big QA corpora are loaded once.
    for task in args.tasks:
        for L in args.lengths:
            pf = os.path.join(parts, f"{task}_{L}.jsonl")
            if os.path.exists(pf) and not args.force:
                n = sum(1 for _ in open(pf))
                if n == args.num_samples:
                    print(f"skip {task} L={L} (exists)")
                    continue
            t0 = time.time()
            rows = generate_task(task, L, args.num_samples, counter)
            with open(pf + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(pf + ".tmp", pf)
            ns = [r["n_tokens_templated"] for r in rows]
            print(f"{task:16s} L={L:6d} n={len(rows)} tok min/mean/max={min(ns)}/{sum(ns)/len(ns):.0f}/{max(ns)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # Merge per length, in canonical task order
    for L in args.lengths:
        out = os.path.join(args.out_dir, f"ruler_{L}.jsonl")
        present = [t for t in ALL_TASKS if os.path.exists(os.path.join(parts, f"{t}_{L}.jsonl"))]
        if not present:
            continue
        if len(present) < len(ALL_TASKS):
            print(f"L={L}: merging the {len(present)} generated tasks only: {present}")
        with open(out, "w") as fo:
            for t in present:
                with open(os.path.join(parts, f"{t}_{L}.jsonl")) as fi:
                    fo.write(fi.read())
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
