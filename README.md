# Retrieval and Context Evaluation Lab

Compare keyword, semantic and hybrid retrieval, then measure which source evidence survives a strict context budget. This is a Python evaluation tool with an offline HTML report, not a generated-answer chatbot.

AI-assisted portfolio prototype prepared in November 2026. Run, inspect and extend it before claiming a personal contribution.

## See the result immediately

Open **`reports/click-comparison.html`** in a browser. No server, API key or model installation is needed to inspect the saved results. Select a retrieval method, token budget, packing policy and query to see ranked source excerpts, citations, exact retained context and missed evidence. JSON and Markdown companions contain the same measurements.

The included case study contains **39 real source excerpts from a pinned Click commit and 24 authored queries**. Its input hashes, file/line citations, per-query failures and embedding truncation audit are recorded. The original 12-document synthetic lexical demo remains available separately.

## Reproduce the semantic comparison

Use Python 3.11–3.13 in a project virtual environment. Commands run from this directory; on Windows use `.venv\Scripts\python.exe` after creating the environment if preferred.

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install ".[semantic]"
python -m retrieval_lab.download_model models/minilm
python -m retrieval_lab.compare --model-dir models/minilm
```

The explicit model download is approximately 91 MB. Both public files are pinned to a revision and SHA-256 verified before use. Inference runs locally on CPU and never downloads files. No model repository Python code is executed. Model weights and installed dependencies are excluded from the project archive; the saved report is portable.

Direct dependencies: NumPy 2.3.5, ONNX Runtime 1.23.2, Tokenizers 0.22.2. A wheel is included in `dist/`; use `pip install "dist/retrieval_eval_lab-0.2.0-py3-none-any.whl[semantic]"` as an alternative to installing the source.

```sh
python -m retrieval_lab.compare --model-dir models/minilm --k 5 --budgets 64 128 256 512 1024 --output reports/my-run.json
python -m retrieval_lab.compare --model-dir models/minilm --lexical-only --output reports/lexical-wordpiece.json
```

The latter skips dense inference but uses the same verified WordPiece tokenizer for budget accounting. The original dependency-free, **character-budget** demo is separate:

```sh
python -m retrieval_lab
python -m retrieval_lab --k 2 --max-chars 160 --output reports/small-context.json
```

## What is compared

| Component | Implementation |
| --- | --- |
| Overlap | Count unique query terms found in an excerpt; deterministic ties by ID. |
| BM25 | k1=1.5, b=0.75; case and identifier normalization. |
| Dense | Pinned all-MiniLM-L6-v2 ONNX model, attention-masked mean pooling, L2 normalization, cosine similarity. |
| Hybrid | Reciprocal rank fusion of full-corpus BM25 and dense rankings with fixed constant 60. |
| Prefix packing | Keep a prefix of the ranked context up to the exact WordPiece limit; mark partial chunks. |
| Whole-chunk packing | Consider top-k hits in rank order, skip excerpts that cannot fit completely, and try later hits. |

No relevance labels enter ranking or packing, and no method-specific parameter search was performed. Recall@k, MRR@k, nDCG@k and hit rate are scored first. Recall is scored again using only complete chunks after packing. All citation headers and separators are counted by tokenizing the complete assembled string.

**These are MiniLM WordPiece context-only budgets**, not counts for an unspecified chat model. They exclude system/user prompts and special tokens. The embedding encoder separately limits inputs to 256 tokens including its special tokens; 19 source excerpts exceed that limit and are explicitly listed as truncated. The displayed source and packed text remain the original cited excerpts.

## Tests and evidence

```sh
python -m unittest discover -s tests -v
# PowerShell: enable the four downloaded-model integration tests
$env:RETRIEVAL_MODEL_DIR = (Resolve-Path models/minilm).Path
python tools/run_validation.py
```

Without `RETRIEVAL_MODEL_DIR`, four integration tests are skipped. With the downloaded model, the suite covers actual embeddings, Unicode token boundaries, model truncation and an end-to-end CLI run, alongside known metric values, label isolation, checksum failures, source provenance and safe HTML embedding. `reports/test-results.json` records the local result. GitHub Actions is configured for Python 3.11–3.13 but has not run remotely.

`tools/verify_evidence.py --model-dir models/minilm --click-checkout /path/to/click` independently recounts saved contexts and compares all source excerpts. Optional `node tools/browser_check.cjs` checks all 960 report states; it requires a separately installed Playwright package and Chromium browser. `CHROME_EXECUTABLE` can point to an installed Chrome, and `PLAYWRIGHT_MODULE` can point to an existing Playwright package. This check does not use a signed-in browser profile.

To recreate the corpus from an unchanged public checkout at commit `2247b35ea1c47c727d7a06e51fa280e12a863ff6`:

```sh
python tools/build_click_dataset.py /path/to/click
```

The extractor reads AST locations without importing Click. It selects prefixes of documented public functions/methods in four files, capped at 24 source lines and 1,400 characters. `data/click-manifest.json` records exact source and corpus hashes. The Click BSD-3-Clause notice is retained in `data/CLICK_LICENSE.txt`.

## Reading the measurements

On this authored set, dense search has higher recall than BM25, while BM25 has higher MRR than dense. Fusion does not win every metric. Small budgets discard substantial evidence, and whole-chunk packing can leave space unused. The report exposes these tradeoffs instead of declaring a universal winner.

The relevance labels identify intended helper excerpts and may omit other useful implementations. They are not independently reviewed ground truth. Partial chunks receive zero complete-chunk recall even when they contain useful text. There is no training/test split or claim of an independent benchmark, answer correctness, production accuracy or guaranteed internship selection.

See `EXPLAIN_THE_PROJECT.md` for a demonstration script, interview questions and a suggested personal contribution; see `VALIDATION.md` for local checks and limits.

## Primary references

Original project code is available under the [MIT license](LICENSE). Click source excerpts retain their BSD-3-Clause notice; model weights are downloaded separately under the model's own license.

- [MiniLM model card and pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ONNX Runtime inference API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [Tokenizers source, pinned release](https://github.com/huggingface/tokenizers/tree/v0.22.2)
- [Click source at the dataset commit](https://github.com/pallets/click/tree/2247b35ea1c47c727d7a06e51fa280e12a863ff6)
- [Information retrieval evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
