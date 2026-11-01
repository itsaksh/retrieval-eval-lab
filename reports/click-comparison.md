# Retrieval and context evaluation

39 source chunks | 24 authored queries | k=5

Tokenizer: MiniLM WordPiece (no special tokens; not a chat-model tokenizer)

| Method | Recall@k | MRR@k | nDCG@k | Hit rate@k |
| --- | ---: | ---: | ---: | ---: |
| overlap | 0.792 | 0.599 | 0.644 | 0.792 |
| bm25 | 0.875 | 0.743 | 0.777 | 0.875 |
| dense | 0.938 | 0.687 | 0.745 | 0.958 |
| hybrid_rrf | 0.917 | 0.750 | 0.798 | 0.917 |

## Context budget sweep

| Method | Budget | Policy | Packed recall | Mean tokens | Mean reduction |
| --- | ---: | --- | ---: | ---: | ---: |
| overlap | 64 | prefix | 0.000 | 64.0 | 95.3% |
| overlap | 64 | whole_chunks | 0.000 | 0.0 | 100.0% |
| overlap | 128 | prefix | 0.042 | 128.0 | 90.6% |
| overlap | 128 | whole_chunks | 0.042 | 8.9 | 99.2% |
| overlap | 256 | prefix | 0.167 | 256.0 | 81.2% |
| overlap | 256 | whole_chunks | 0.188 | 169.6 | 87.3% |
| overlap | 512 | prefix | 0.521 | 512.0 | 62.4% |
| overlap | 512 | whole_chunks | 0.542 | 367.2 | 72.7% |
| overlap | 1024 | prefix | 0.646 | 1024.0 | 24.7% |
| overlap | 1024 | whole_chunks | 0.646 | 903.8 | 33.7% |
| bm25 | 64 | prefix | 0.000 | 64.0 | 95.1% |
| bm25 | 64 | whole_chunks | 0.000 | 0.0 | 100.0% |
| bm25 | 128 | prefix | 0.062 | 128.0 | 90.2% |
| bm25 | 128 | whole_chunks | 0.062 | 18.0 | 98.4% |
| bm25 | 256 | prefix | 0.229 | 256.0 | 80.4% |
| bm25 | 256 | whole_chunks | 0.229 | 177.6 | 86.3% |
| bm25 | 512 | prefix | 0.646 | 512.0 | 60.8% |
| bm25 | 512 | whole_chunks | 0.646 | 393.6 | 69.4% |
| bm25 | 1024 | prefix | 0.875 | 1017.8 | 22.3% |
| bm25 | 1024 | whole_chunks | 0.875 | 904.9 | 30.6% |
| dense | 64 | prefix | 0.000 | 64.0 | 94.9% |
| dense | 64 | whole_chunks | 0.000 | 0.0 | 100.0% |
| dense | 128 | prefix | 0.042 | 128.0 | 89.9% |
| dense | 128 | whole_chunks | 0.062 | 36.6 | 96.8% |
| dense | 256 | prefix | 0.125 | 256.0 | 79.7% |
| dense | 256 | whole_chunks | 0.146 | 193.6 | 84.6% |
| dense | 512 | prefix | 0.521 | 512.0 | 59.5% |
| dense | 512 | whole_chunks | 0.542 | 398.2 | 68.1% |
| dense | 1024 | prefix | 0.854 | 1021.7 | 19.2% |
| dense | 1024 | whole_chunks | 0.896 | 905.9 | 28.0% |
| hybrid_rrf | 64 | prefix | 0.000 | 64.0 | 95.1% |
| hybrid_rrf | 64 | whole_chunks | 0.000 | 0.0 | 100.0% |
| hybrid_rrf | 128 | prefix | 0.042 | 128.0 | 90.1% |
| hybrid_rrf | 128 | whole_chunks | 0.062 | 22.5 | 97.9% |
| hybrid_rrf | 256 | prefix | 0.167 | 256.0 | 80.2% |
| hybrid_rrf | 256 | whole_chunks | 0.188 | 187.8 | 85.2% |
| hybrid_rrf | 512 | prefix | 0.625 | 512.0 | 60.4% |
| hybrid_rrf | 512 | whole_chunks | 0.667 | 390.0 | 69.4% |
| hybrid_rrf | 1024 | prefix | 0.917 | 1017.8 | 21.6% |
| hybrid_rrf | 1024 | whole_chunks | 0.917 | 904.3 | 30.0% |

## Retrieval misses and partial misses

- overlap: c07 (0.00), c09 (0.00), c15 (0.00), c17 (0.00), c18 (0.00)
- bm25: c07 (0.00), c15 (0.00), c17 (0.00)
- dense: c17 (0.00), c18 (0.50)
- hybrid_rrf: c07 (0.00), c17 (0.00)

## Interpretation

Small authored relevance set, not an independent benchmark. Complete-chunk recall only; no answer generation. WordPiece context-only budgets exclude prompts and special tokens.

Labels identify the intended helper excerpts, not every potentially useful implementation. A partially truncated chunk receives zero complete-chunk credit. The whole-chunk policy skips oversized chunks without summarizing or generating text. Neither policy sees relevance labels.

Embedding-truncated documents: 19; queries: 0.
The same fixed queries are used for every method; no per-method parameter search was performed. Scores are descriptive on this small authored set, not evidence of production quality or selection likelihood.

Input provenance and exact per-query context are in the JSON and HTML companions.
