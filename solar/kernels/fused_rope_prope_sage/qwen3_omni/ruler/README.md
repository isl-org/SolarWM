# RULER long-context data for Qwen3-Omni-30B-A3B

This is NVIDIA RULER (https://github.com/NVIDIA/RULER), cloned into `../third_party/RULER`. It has 12 synthetic tasks × 5 lengths × 20 samples, which is 1200 samples.

- Tasks: niah_single_1/2/3, niah_multikey_1/2, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1 (SQuAD), qa_2 (HotpotQA). The task args are the defaults in `scripts/synthetic.yaml`. Templates, answer prefixes and `tokens_to_generate` come from `scripts/data/synthetic/constants.py`.
- Lengths: 2048, 8192, 16384, 32768, 63488. Length is counted with the Qwen3-Omni tokenizer (`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`) after `apply_chat_template([{"role":"user","content":input}], add_generation_prompt=True)`. Every sample lies in `[floor(0.97*L), L]`. The 64K bucket is capped at 63488 so that the prompt plus generated tokens stays under 65536.

## How it was generated
`generate.py` ports the RULER generators (`niah.py`, `variable_tracking.py`, `common_words_extraction.py`, `freq_words_extraction.py`, `qa.py`) into one process. The prompt text, needles, haystacks, few-shot/ICL examples and gold answers follow RULER. Two things differ:
- **Length fitting.** RULER does one binary search per task on `len(tok(input)) + tokens_to_generate <= max_seq_length`. Here each sample has its own search over the haystack size (words, sentences, docs or list entries) on the chat-templated Qwen token count, with the target `L` itself; nothing is subtracted for generation. If a sample cannot land in the window, the random draw is redone with a new sub-seed. In practice this only happens for qa at 2048, where whole documents are too coarse. `haystack_units` and `seed` are stored with each sample.
- **qa_2 at 2048.** HotpotQA questions 7 and 13 do not fit in 2048 tokens even with only their own 10 paragraphs. Upstream RULER would loop forever on them. They are skipped and the next unused question is taken, so qa_2@2048 uses questions 0-6, 8-12 and 14-21. `source_qa_index` (qa only) records which question each sample uses. Every other length uses questions 0-19, as RULER does.
- **Seeding.** Each sample uses an RNG seeded from md5(`42|task|L|index|attempt`), so the run is deterministic. RULER's fixed-seed shuffles (`random.Random(42)`) are kept as they are.

The RULER inputs come from RULER's own scripts and live in `../third_party/RULER/scripts/data/synthetic/json/`:
`squad.json` and `hotpotqa.json` (from `download_qa_dataset.sh`) and `PaulGrahamEssays.json`. The essays are fetched by `download_essays.py`, which uses the same logic as RULER's `download_paulgraham_essay.py` but retries each URL. The upstream script silently dropped essays when the proxy reset connections. All 218 essays were fetched.

## Regenerate
```bash
B=/home/gta/omni_bench
git clone https://github.com/NVIDIA/RULER $B/third_party/RULER
$B/env.sh pip install nltk wonderwords tenacity html2text beautifulsoup4
NLTK_ALLOW_PROXIED_URLOPEN=1 $B/env.sh python -I -c "import nltk; [nltk.download(p, download_dir='/home/gta/omni_env/nltk_data') for p in ('punkt','punkt_tab')]"
cd $B/third_party/RULER/scripts/data/synthetic/json && bash download_qa_dataset.sh
cd $B/ruler
$B/env.sh python -I download_essays.py $B/third_party/RULER/scripts/data/synthetic/json
$B/env.sh python generate.py            # CPU only, about 1.4 GB RAM; parts are cached in data/parts/ (--force to redo)
$B/env.sh python validate.py --recount  # token table, window check, gold-in-input check
```

## Files
- `data/ruler_{L}.jsonl` holds the 240 lines for one length, in task order.
- `data/parts/{task}_{L}.jsonl` holds the per-task pieces that the merged files are built from.
- `score.py` is the scorer.

## Schema (one JSON object per line)
| field | meaning |
|---|---|
| `task` | task name, e.g. `niah_single_1` |
| `index` | 0..19 within (task, length) |
| `length_bucket` | target L |
| `input` | raw user message, without the chat template and without the answer prefix |
| `n_tokens_templated` | Qwen3-Omni token count of `apply_chat_template([{"role":"user","content":input}], add_generation_prompt=True)` |
| `outputs` | list of gold answer strings |
| `max_new_tokens` | RULER `tokens_to_generate` (niah 128, vt 30, cwe 120, fwe 50, qa 32) |
| `answer_prefix` | RULER answer prefix, e.g. `" Answer:"`. RULER puts it after the assistant tag as a prefill (`input + answer_prefix`). Using it is optional and adds 2–40 tokens. |
| `haystack_units`, `seed` | fitting info that lets you reproduce a sample |
| `source_qa_index` | qa only: index into RULER's QA list |

Worst-case `n_tokens_templated + answer_prefix + max_new_tokens` at 63488 is 63521 (fwe), which is below 65536.
`validate.py` reports gold-not-in-input for 8 qa_2 samples. These are HotpotQA yes/no comparison questions, so this is expected.

## Scoring
```python
from score import score            # per-sample value in [0, 1]
score("niah_multivalue", pred, row["outputs"])
```
`string_match_all` is used for niah, vt, cwe and fwe: the fraction of gold strings that appear in the prediction, case-insensitive. `string_match_part` is used for qa: 1 if any gold string appears. Predictions are cleaned the same way as RULER's `postprocess_pred`.
CLI: `python score.py preds.jsonl [--pred_field pred] [--out summary.json]`. Each line needs `task`, `length_bucket`, `index` and `pred`; `outputs` is optional and is looked up in `data/` if missing. The CLI prints a task × length table (mean × 100) and the average over tasks.
