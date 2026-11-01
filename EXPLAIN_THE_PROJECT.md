# Understand and demonstrate this project

This prototype was prepared with AI assistance in November 2026. Akshaya should run it, inspect the code and implement a change she can explain before presenting a personal contribution. No GitHub publication, production use, external users or hiring outcome is claimed.

## A two-minute demonstration

1. Open `reports/click-comparison.html`. It works offline without a server or model download.
2. Compare BM25, dense and hybrid retrieval in the summary. Notice that higher recall does not imply better first-result ranking.
3. Choose a query and switch between 128, 512 and 1024 WordPiece tokens. The exact retained context, partial chunks and omitted chunks are visible.
4. Compare prefix truncation with whole chunks. Explain why a partial chunk earns zero **complete-chunk** recall even though it may contain useful text.
5. Follow one source citation to the pinned Click commit, or inspect its excerpt locally in the report.
6. Inspect `VALIDATION.md` and the JSON provenance. Explain the model's 256-token embedding input limit and its truncation audit.

## Questions to be able to answer

- **What is BM25?** A term-based ranking function using term frequency, inverse document frequency and document-length normalization. This implementation fixes k1=1.5 and b=0.75.
- **What does the dense method do?** The pinned MiniLM model produces token representations in ONNX Runtime on CPU. The code takes an attention-masked mean, normalizes the resulting 384-dimensional vector, and ranks by cosine similarity. No generative LLM or vector database is required for 39 chunks.
- **Why RRF?** BM25 scores and cosine scores have different scales. Reciprocal rank fusion combines their rank positions using 1/(60+rank), with a fixed constant and full-corpus rankings.
- **What is recall@5?** The fraction of a query's intended relevant chunks in its first five results. MRR uses the rank of the first relevant result. nDCG also accounts for the positions of multiple relevant chunks.
- **What gets counted?** The complete assembled context, including citation headers and separators, using MiniLM's WordPiece tokenizer with truncation and padding disabled. It is not the tokenizer of a hypothetical chat model, and does not include a system prompt or special tokens.
- **What is retained evidence?** The fraction of labelled chunks that remain complete after packing. This is a deliberately strict chunk-level proxy, not generated-answer correctness.
- **Where are labels used?** Only in evaluation after retrieval and context selection. Changing labels changes scores but cannot change the ranks or selected text; a regression test verifies this separation.
- **Why are there failures?** Some helpers have similar documentation, some queries paraphrase the source, and many chunks are too large for small budgets. Nineteen source chunks exceed the encoder's 256-token input limit. The report records this limitation rather than hiding it.

## A useful personal contribution

Add an adaptive source chunker that respects function boundaries and the encoder's input limit. Freeze a new query set before comparing policies. Preserve source-line mappings and show whether smaller chunks improve complete-chunk recall without losing the information a reviewer needs. Keep the current report as the baseline, and document exactly what changed.

## Resume wording after review and contribution

“Developed an AI-assisted Python retrieval evaluation prototype comparing BM25, local MiniLM embeddings and rank fusion on 39 pinned Click excerpts and 24 authored queries; measured recall/MRR and evidence retained under exact WordPiece context budgets.”

Use only the part of this statement that accurately describes Akshaya's own reviewed work. The figures describe the included demo, not a production benchmark.
