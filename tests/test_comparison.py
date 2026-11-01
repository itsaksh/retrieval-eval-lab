import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from retrieval_lab.experiment import (ComparisonRetriever, citation, evaluate_comparison,
                                      ndcg, pack_tokens, reciprocal_rank_fusion)
from retrieval_lab.model import mean_pool_normalize, verify_model
from retrieval_lab.report import html_report, markdown_report

ROOT = Path(__file__).resolve().parents[1]
DOCS = [{"id": "a", "text": "Python AST source"}, {"id": "b", "text": "Git branch review"}]
QUERIES = [{"id": "q", "query": "Python source", "relevant": ["a"]}]


class CharacterCounter:
    # Deliberate test double, never labelled as model-token counting in a report.
    name = "test-only character counter"
    def count(self, text):
        return len(text)


class ComparisonTests(unittest.TestCase):
    def test_bm25_scores_match_across_python_hash_seeds(self):
        code = ("import json; from retrieval_lab.core import Retriever; "
                "docs=[{'id':str(i),'text':' '.join(['alpha']*(i+1)+['beta','gamma','delta','epsilon'][:i%4+1])} for i in range(12)]; "
                "print(json.dumps(Retriever(docs).search('alpha beta gamma delta epsilon',12)))")
        results = []
        for seed in ("1", "123"):
            results.append(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT,
                           env={**os.environ, "PYTHONHASHSEED": seed}, text=True, timeout=15))
        self.assertEqual(results[0], results[1])

    def test_fusion_uses_ranks_not_incomparable_raw_scores(self):
        a, b = {"id": "a", "text": "a", "score": 999}, {"id": "b", "text": "b", "score": .01}
        hits = reciprocal_rank_fusion([[a, b], [b]])
        self.assertEqual([h["id"] for h in hits], ["b", "a"])
        self.assertAlmostEqual(hits[0]["score"], 1/62+1/61)

    def test_fusion_ties_and_duplicates(self):
        a, b = {"id": "a"}, {"id": "b"}
        self.assertEqual([h["id"] for h in reciprocal_rank_fusion([[b], [a]])], ["a", "b"])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[a, a]])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[a]], True)

    def test_known_ndcg_value(self):
        self.assertAlmostEqual(ndcg(["x", "a", "b"], {"a", "b"}, 3),
                               (1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3)))

    def test_whole_chunk_skips_long_hit_preserves_short_hit(self):
        hits = [{"id": "big", "text": "x"*80}, {"id": "a", "text": "ok"}]
        packed = pack_tokens(hits, 8, CharacterCounter())
        self.assertEqual(packed["text"], "[a]\nok")
        self.assertEqual(packed["selected_ids"], ["a"])
        self.assertEqual(packed["skipped_ids"], ["big"])

    def test_exact_boundary_counts_citations_separators_and_unicode(self):
        hits = [{"id": "a", "text": "é"}, {"id": "b", "text": "字"}]
        self.assertEqual(pack_tokens(hits, 12, CharacterCounter())["selected_ids"], ["a", "b"])
        self.assertEqual(pack_tokens(hits, 11, CharacterCounter())["selected_ids"], ["a"])
        self.assertEqual(pack_tokens(hits, 0, CharacterCounter())["text"], "")

    def test_prefix_marks_partial_chunk_without_giving_full_credit(self):
        hits = [{"id": "big", "text": "x"*20}, {"id": "a", "text": "ok"}]
        packed = pack_tokens(hits, 12, CharacterCounter(), "prefix")
        self.assertEqual(packed["text"], "[big]\nxxxxxx")
        self.assertEqual(packed["selected_ids"], [])
        self.assertEqual(packed["partial_ids"], ["big"])
        self.assertEqual(packed["skipped_ids"], ["a"])

    def test_empty_hits_do_not_divide_by_zero(self):
        packed = pack_tokens([], 100, CharacterCounter())
        self.assertEqual(packed["token_reduction_fraction"], 0)

    def test_packing_rejects_invalid_budgets_and_duplicate_ids(self):
        for budget in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                pack_tokens(DOCS, budget, CharacterCounter())
        with self.assertRaises(ValueError):
            pack_tokens([DOCS[0], DOCS[0]], 100, CharacterCounter())

    def test_source_location_is_part_of_counted_context(self):
        hit = {"id": "x", "text": "pass", "source": {"path": "a.py", "start_line": 2, "end_line": 2}}
        self.assertEqual(citation(hit), "[x | a.py:2-2]\npass")
        self.assertEqual(pack_tokens([hit], 10, CharacterCounter())["selected_ids"], [])

    def test_labels_do_not_change_rankings_or_context(self):
        first = evaluate_comparison(DOCS, QUERIES, CharacterCounter(), k=2, budgets=[10, 100])
        other = copy.deepcopy(QUERIES)
        other[0]["relevant"] = ["b"]
        second = evaluate_comparison(DOCS, other, CharacterCounter(), k=2, budgets=[10, 100])
        for method in first["methods"]:
            x, y = first["methods"][method]["queries"][0], second["methods"][method]["queries"][0]
            self.assertEqual(x["hits"], y["hits"])
            self.assertNotEqual(x["recall_at_k"], y["recall_at_k"])
            for px, py in zip(x["packs"], y["packs"]):
                self.assertEqual(px["text"], py["text"])
                self.assertEqual(px["selected_ids"], py["selected_ids"])

    def test_document_labels_and_arbitrary_metadata_are_excluded(self):
        retriever = ComparisonRetriever([{**DOCS[0], "relevant": ["secret"], "hidden": "keyword"}])
        self.assertEqual(set(retriever.documents["a"]), {"id", "text"})
        self.assertEqual(retriever.search("keyword", 1, "bm25"), [])

    def test_aggregate_scores_and_undefined_retention(self):
        report = evaluate_comparison(DOCS, QUERIES, CharacterCounter(), k=1, budgets=[0, 100])
        bm25 = report["methods"]["bm25"]
        self.assertEqual(bm25["mean_recall_at_k"], 1)
        self.assertEqual(bm25["packing"][0]["mean_packed_recall"], 0)
        self.assertEqual(bm25["packing"][2]["mean_packed_recall"], 1)
        missing = [{"id": "q", "query": "unmatched", "relevant": ["a"]}]
        report = evaluate_comparison(DOCS, missing, CharacterCounter(), k=1, budgets=[100])
        self.assertIsNone(report["methods"]["bm25"]["packing"][0]["mean_evidence_retention_when_retrieved"])

    def test_bad_queries_and_settings_rejected(self):
        for queries in ([], [None], [QUERIES[0], QUERIES[0]], [{"id": "q", "query": "a", "relevant": ["missing"]}]):
            with self.assertRaises(ValueError):
                evaluate_comparison(DOCS, queries, CharacterCounter())
        for budgets in ([], [1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                evaluate_comparison(DOCS, QUERIES, CharacterCounter(), budgets=budgets)
        with self.assertRaises(ValueError):
            evaluate_comparison(DOCS, QUERIES, CharacterCounter(), k=True)

    def test_report_data_cannot_terminate_script_block(self):
        docs = [{"id": "x", "text": "</script><script>alert('x')</script> Python"}]
        report = evaluate_comparison(docs, [{"id": "q", "query": "Python", "relevant": ["x"]}], CharacterCounter())
        html = html_report(report)
        self.assertNotIn("<script>alert", html)
        payload = html.split('<script type="application/json" id="data">')[1].split("</script>")[0]
        self.assertEqual(json.loads(payload), report)
        self.assertIn("bm25", markdown_report(report))

    def test_included_dataset_provenance_and_labels(self):
        data = ROOT / "data/click-documents.jsonl"
        manifest = json.loads((ROOT/"data/click-manifest.json").read_text())
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), manifest["documents_sha256"])
        docs = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(docs), 39)
        for doc in docs:
            source = doc["source"]
            self.assertEqual(source["commit"], manifest["commit"])
            self.assertEqual(len(doc["text"].splitlines()), source["end_line"]-source["start_line"]+1)
        queries = [json.loads(line) for line in (ROOT/"data/click-queries.jsonl").read_text().splitlines()]
        self.assertEqual(len(queries), 24)
        evaluate_comparison(docs, queries, CharacterCounter(), k=5, budgets=[100])

    def test_model_files_are_verified_not_silently_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny"
            path.write_bytes(b"valid")
            with patch("retrieval_lab.model.FILES", {"tiny": hashlib.sha256(b"valid").hexdigest()}):
                verify_model(directory)
                path.write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    verify_model(directory)


class PoolingTests(unittest.TestCase):
    def test_padding_is_masked_before_normalization(self):
        import numpy as np
        hidden = np.array([[[2., 0.], [0., 2.], [1000., 1000.]]])
        pooled = mean_pool_normalize(hidden, np.array([[1, 1, 0]]))
        np.testing.assert_allclose(pooled, [[1/math.sqrt(2), 1/math.sqrt(2)]], rtol=1e-6)

    def test_invalid_model_outputs_are_rejected(self):
        import numpy as np
        for hidden, mask in [(np.zeros((1, 2)), [[1, 1]]),
                             (np.ones((1, 2, 3)), [[0, 0]]),
                             (np.ones((1, 2, 3))*float("nan"), [[1, 1]]),
                             (np.zeros((1, 2, 3)), [[1, 1]])]:
            with self.assertRaises(ValueError):
                mean_pool_normalize(hidden, mask)


@unittest.skipUnless(os.environ.get("RETRIEVAL_MODEL_DIR"), "set RETRIEVAL_MODEL_DIR for pinned-model integration tests")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from retrieval_lab.model import MiniLMEncoder, WordPieceCounter
        cls.directory = Path(os.environ["RETRIEVAL_MODEL_DIR"])
        cls.encoder = MiniLMEncoder(cls.directory)
        cls.counter = WordPieceCounter(cls.directory)

    def test_real_embeddings_are_unit_length_and_semantic(self):
        import numpy as np
        vectors, truncated = self.encoder.encode(["A puppy is playing.", "A dog plays in the garden.", "Installing a database server."])
        self.assertEqual(vectors.shape, (3, 384))
        self.assertEqual(truncated, [])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        self.assertGreater(float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2]))

    def test_real_token_budget_is_exact_including_unicode_and_citations(self):
        hits = [{"id": "a", "text": "Café naïve 🚀 नमस्ते Python!"}, {"id": "b", "text": "another file"}]
        for budget in range(25):
            for strategy in ("prefix", "whole_chunks"):
                packed = pack_tokens(hits, budget, self.counter, strategy)
                self.assertEqual(packed["tokens"], self.counter.count(packed["text"]))
                self.assertLessEqual(packed["tokens"], budget)
                if strategy == "prefix":
                    self.assertTrue("\n\n".join(citation(h) for h in hits).startswith(packed["text"]))

    def test_long_embedding_inputs_have_explicit_truncation_audit(self):
        _, truncated = self.encoder.encode(["short text", "word "*300])
        self.assertEqual(truncated, [1])

    def test_all_four_methods_and_real_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run([sys.executable, "-m", "retrieval_lab.compare", "--model-dir", str(self.directory),
                            "--budgets", "128", "512", "--output", str(output)], cwd=ROOT,
                           capture_output=True, text=True, timeout=60, check=True)
            report = json.loads(output.read_text())
            self.assertEqual(set(report["methods"]), {"overlap", "bm25", "dense", "hybrid_rrf"})
            self.assertEqual(report["queries"], 24)
            self.assertEqual(report["embedding"]["dimensions"], 384)
            self.assertTrue(output.with_suffix(".html").is_file())
            for result in report["methods"].values():
                for row in result["queries"]:
                    for packed in row["packs"]:
                        self.assertLessEqual(packed["tokens"], packed["budget_tokens"])


if __name__ == "__main__":
    unittest.main()
