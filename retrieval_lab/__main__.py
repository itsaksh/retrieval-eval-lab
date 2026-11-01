import argparse
import hashlib
import json
from pathlib import Path
from .core import evaluate


def read_jsonl(path):
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"Invalid JSON at {path.name}:{number}") from exc
    return records


def main():
    parser = argparse.ArgumentParser(description="Compare lexical retrieval and context selection")
    parser.add_argument("--documents", type=Path, default=Path("data/documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("reports/demo.json"))
    args = parser.parse_args()
    try:
        result = evaluate(read_jsonl(args.documents), read_jsonl(args.queries), args.k, args.max_chars)
        result["inputs"] = {"documents_sha256": hashlib.sha256(args.documents.read_bytes()).hexdigest(),
                            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        lines = ["# Retrieval evaluation results", "", f"Documents: {result['documents']} | Queries: {result['queries']} | k: {args.k} | Character budget: {args.max_chars}",
                 "", "| Method | Recall at k | MRR at k | Hit rate at k | Recall after packing |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, values in result["methods"].items():
            lines.append(f"| {name} | {values['mean_recall_at_k']:.3f} | {values['mrr_at_k']:.3f} | {values['hit_rate_at_k']:.3f} | {values['mean_packed_recall_at_k']:.3f} |")
        lines += ["", "The included corpus is tiny and synthetic. Results describe these fixtures only; they do not establish production retrieval quality.",
                  "Character counts are exact string lengths, not model-token counts. Metrics do not evaluate generated-answer correctness."]
        args.output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("\n".join(lines))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
