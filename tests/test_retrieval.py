import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from retrieval_lab.core import Retriever, evaluate, metrics, pack_context, tokens

ROOT = Path(__file__).resolve().parents[1]
DOCS = [{"id": "a", "text": "Python AST functions source"},
        {"id": "b", "text": "Git review and tests"},
        {"id": "c", "text": "Python MCP tools server"}]


class RetrievalTests(unittest.TestCase):
    def test_identifier_tokenization(self):
        self.assertEqual(tokens("getSymbol_context MCP"), ["get", "symbol", "context", "mcp"])

    def test_search_ranks_relevant_document(self):
        r = Retriever(DOCS)
        for method in ("bm25", "overlap"):
            self.assertEqual(r.search("AST functions", method=method)[0]["id"], "a")

    def test_empty_and_unmatched_query_do_not_fabricate_hits(self):
        r = Retriever(DOCS)
        self.assertEqual(r.search(""), [])
        self.assertEqual(r.search("unicorn"), [])

    def test_empty_documents_handle_zero_average_length(self):
        self.assertEqual(Retriever([{"id": "empty", "text": ""}]).search("python"), [])

    def test_stable_tie_order(self):
        r = Retriever([{"id": "b", "text": "same"}, {"id": "a", "text": "same"}])
        self.assertEqual([h["id"] for h in r.search("same")], ["a", "b"])

    def test_exact_metric_values(self):
        scores = metrics(["a", "b", "c"], {"b", "d"}, 3)
        self.assertEqual(scores, {"recall_at_k": .5, "reciprocal_rank_at_k": .5, "hit_at_k": 1.0})
        self.assertEqual(metrics(["a", "b"], {"b"}, 1)["hit_at_k"], 0)

    def test_duplicate_ids_and_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            Retriever(DOCS+[DOCS[0]])
        with self.assertRaises(ValueError):
            evaluate(DOCS, [{"id": "q", "query": "test", "relevant": ["missing"]}])
        with self.assertRaises(ValueError):
            metrics(["a", "a"], {"a"}, 2)

    def test_empty_queries_and_invalid_k_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(DOCS, [])
        for k in (0, -1, True):
            with self.assertRaises(ValueError):
                Retriever(DOCS).search("python", k=k)

    def test_packer_keeps_budget_and_citations(self):
        hits = [{"id": "long", "text": "x"*100}, {"id": "a", "text": "short"}]
        packed = pack_context(hits, 20)
        self.assertEqual(packed["selected_ids"], ["a"])
        self.assertEqual(packed["skipped_ids"], ["long"])
        self.assertEqual(packed["text"], "[a]\nshort")
        self.assertLessEqual(packed["characters"], 20)

    def test_packer_accounts_for_separators_and_unicode(self):
        hits = [{"id": "a", "text": "\u00e9"}, {"id": "b", "text": "x"}]
        self.assertEqual(pack_context(hits, 12)["characters"], 12)
        self.assertEqual(pack_context(hits, 11)["selected_ids"], ["a"])
        self.assertEqual(pack_context(hits, 0)["selected_ids"], [])

    def test_recall_after_context_selection_is_measured(self):
        report = evaluate(DOCS, [{"id": "q", "query": "AST", "relevant": ["a"]}], max_chars=1)
        self.assertEqual(report["methods"]["bm25"]["mean_recall_at_k"], 1)
        self.assertEqual(report["methods"]["bm25"]["mean_packed_recall_at_k"], 0)

    def test_cli_writes_reproducible_machine_readable_results(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"result.json"
            subprocess.run([sys.executable, "-m", "retrieval_lab", "--output", str(out)],
                           cwd=ROOT, capture_output=True, text=True, timeout=15, check=True)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["queries"], 16)
            self.assertEqual(set(report["methods"]), {"overlap", "bm25"})
            self.assertEqual(len(report["inputs"]["queries_sha256"]), 64)
            self.assertTrue(out.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
