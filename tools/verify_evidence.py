"""Independently check saved contexts and their original source-line mappings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_lab.model import WordPieceCounter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--click-checkout", required=True, type=Path)
    args = parser.parse_args()
    counter = WordPieceCounter(args.model_dir)
    docs = [json.loads(line) for line in (ROOT/"data/click-documents.jsonl").read_text(encoding="utf-8").splitlines()]
    for doc in docs:
        s = doc["source"]
        lines = (args.click_checkout/s["path"]).read_text(encoding="utf-8").splitlines()
        assert "\n".join(lines[s["start_line"]-1:s["end_line"]]) == doc["text"], doc["id"]
    report = json.loads((ROOT/"reports/click-comparison.json").read_text(encoding="utf-8"))
    checked = 0
    for method in report["methods"].values():
        for row in method["queries"]:
            for context in row["packs"]:
                actual = counter.count(context["text"])
                assert actual == context["tokens"] <= context["budget_tokens"]
                for key in context["selected_ids"]:
                    assert any(h["id"] == key and h["text"] in context["text"] for h in row["hits"])
                checked += 1
    result = {"source_excerpts_match_pinned_checkout": len(docs), "contexts_recounted_with_real_tokenizer": checked,
              "all_contexts_within_budget": True}
    (ROOT/"reports/evidence-check.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
