# Local validation — 1 November 2026

Verified with Python 3.12.14 on Windows. The public [GitHub Actions run](https://github.com/itsaksh/retrieval-eval-lab/actions/runs/41320903660) also passed on Python 3.11, 3.12 and 3.13 on Linux. Hosted CI skips the four downloaded-model integration tests; all 35 ran locally with the pinned model.

| Check | Result |
| --- | --- |
| Unit and real-model integration suite | 35 passed, 0 failed, 0 skipped. |
| End-to-end semantic CLI | Four methods, 39 Click excerpts, 24 queries, five budgets and two packing policies. |
| Saved-context verification | All 960 contexts independently recounted with the real WordPiece tokenizer; every context within its stated budget. |
| Source-line verification | All 39 saved excerpts exactly match the cited lines in the pinned public Click checkout. |
| Browser interaction | All 960 method/budget/policy/query states displayed the expected context; no JavaScript errors. |
| Layout | Desktop and 390-pixel mobile viewport checked for horizontal overflow; desktop report screenshot visually inspected. |
| Packaging | Wheel built, installed outside the source directory and compared with the source-generated report; source archive checked for integrity. |

Machine-readable records: `reports/test-results.json`, `reports/evidence-check.json`, `reports/browser-check.json`, and `reports/package-check.json`. The HTML browser check used an isolated headless Chrome instance; it did not use an existing signed-in browser session.

## Observed retrieval results

| Method | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: |
| Overlap | 0.792 | 0.599 | 0.644 |
| BM25 | 0.875 | 0.743 | 0.777 |
| MiniLM dense | 0.938 | 0.687 | 0.745 |
| Hybrid RRF | 0.917 | 0.750 | 0.798 |

These describe the included authored labels only. Higher dense recall did not yield higher dense MRR than BM25; fusion did not dominate all metrics. The budget sweep and failures are included in full in the report.

## Limits and issues surfaced

- Nineteen source excerpts exceed the encoder's 256-token input length. Their IDs are recorded; embedding input is truncated while displayed source remains intact. This can affect ranking.
- Very small budgets retain little or no complete evidence. Whole-chunk packing can leave substantial unused capacity. Prefix truncation preserves a partial excerpt but receives no complete-chunk credit for it.
- The labels were authored for the selected source excerpts, not independently adjudicated. Other useful implementations may be unlabelled. No method-specific parameter search or general benchmark claim is made.
- WordPiece counts are context-only, including citation headers and separators. They do not represent chat-model tokens or complete prompt size.
- Tests caught an extractor boundary that ended on a blank source line. The extractor now removes trailing blank lines and preserves the exact inclusive line citation; source comparisons pass.
- Browser checks caught overflow in a narrow viewport. Responsive controls were adjusted and both viewport checks now pass.
- The installed-package comparison caught tiny BM25 floating-point differences across hash seeds. Query terms are now summed in sorted order, with a cross-process regression test; installed and source-generated reports match exactly on this machine.
- The first GitHub run exposed Windows-to-Linux newline normalization in the corpus checksum. The extractor now reads committed Git blobs and writes canonical LF bytes; the provenance test also asserts LF-only corpus storage.

Model revision and file checksums, input SHA-256 hashes, fixed ranking settings, encoder truncations, per-query rankings and exact context text are in `reports/click-comparison.json`. Model weights and dependencies are not packaged in the source archive. The implementation and demo were created with AI assistance; no production deployment, users, GitHub publication or internship selection is claimed.
