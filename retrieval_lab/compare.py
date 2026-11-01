import argparse
import hashlib
import json
from pathlib import Path
from .__main__ import read_jsonl
from .experiment import evaluate_comparison
from .model import MiniLMEncoder, WordPieceCounter
from .report import html_report, markdown_report


def main():
    parser = argparse.ArgumentParser(description="Compare lexical, semantic and hybrid search under actual WordPiece budgets")
    parser.add_argument("--documents", type=Path, default=Path("data/click-documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/click-queries.jsonl"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--lexical-only", action="store_true", help="Skip embedding inference; still uses exact WordPiece counting")
    parser.add_argument("--output", type=Path, default=Path("reports/click-comparison.json"))
    args = parser.parse_args()
    try:
        counter = WordPieceCounter(args.model_dir)
        encoder = None if args.lexical_only else MiniLMEncoder(args.model_dir)
        result = evaluate_comparison(read_jsonl(args.documents), read_jsonl(args.queries), counter, encoder, args.k, args.budgets)
        result["inputs"] = {"documents_file": args.documents.name, "queries_file": args.queries.name,
                            "documents_sha256": hashlib.sha256(args.documents.read_bytes()).hexdigest(),
                            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest()}
        result["settings"] = {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_constant": 60,
                              "fusion_depth": "full corpus", "tuning": "none; fixed settings across authored queries",
                              "packing_credit": "complete chunks only", "tokenizer_provenance": counter.provenance}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
        args.output.with_suffix(".html").write_text(html_report(result), encoding="utf-8")
        print(json.dumps({"report": str(args.output), "documents": result["documents"], "queries": result["queries"],
                          "recall_at_k": {m:r["mean_recall_at_k"] for m,r in result["methods"].items()},
                          "embedding_truncated_documents": len(result["truncated_document_ids"])}, indent=2))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"{exc}\nInstall optional inference packages with: pip install '.[semantic]'\n")


if __name__ == "__main__":
    main()
