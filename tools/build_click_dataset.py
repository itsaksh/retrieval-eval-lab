"""Deterministic, read-only extraction from a pinned public Click checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "2247b35ea1c47c727d7a06e51fa280e12a863ff6"
FILES = ["src/click/utils.py", "src/click/termui.py", "src/click/testing.py", "src/click/decorators.py"]


def build(checkout, output, git="git"):
    actual = subprocess.check_output([git, "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"checkout must be at {COMMIT}")
    dirty = subprocess.check_output([git, "-C", str(checkout), "status", "--porcelain", "--", *FILES, "LICENSE.txt"], text=True)
    if dirty.strip():
        raise ValueError("dataset source files have local changes")
    records, sources = [], {}
    for relative in FILES:
        raw = (checkout / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        tree = ast.parse(text)
        def visit(body, parents=()):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        visit(node.body, (*parents, node.name))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Public functions/methods with documentation; exclude overload stubs.
                    if node.name.startswith("_") or not ast.get_docstring(node):
                        continue
                    name = ".".join((*parents, node.name))
                    end = min(node.end_lineno, node.lineno + 23)
                    while end > node.lineno and len("\n".join(lines[node.lineno-1:end])) > 1400:
                        end -= 1
                    while end > node.lineno and not lines[end-1].strip():
                        end -= 1
                    excerpt = "\n".join(lines[node.lineno-1:end])
                    records.append({"id": f"{Path(relative).stem}.{name}", "text": excerpt,
                                    "source": {"repository": "https://github.com/pallets/click", "commit": COMMIT,
                                               "path": relative, "start_line": node.lineno, "end_line": end,
                                               "symbol_end_line": node.end_lineno,
                                               "url": f"https://github.com/pallets/click/blob/{COMMIT}/{relative}#L{node.lineno}-L{end}"}})
        visit(tree.body)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate symbol ids")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "click-documents.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    (output / "CLICK_LICENSE.txt").write_bytes((checkout / "LICENSE.txt").read_bytes())
    manifest = {"repository": "https://github.com/pallets/click", "commit": COMMIT,
                "source_files_sha256": sources, "documents_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "chunks": len(records), "policy": "One prefix per documented public function/method; at most 24 source lines and 1400 characters; no generated summaries."}
    (output / "click-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"documents": len(records), "ids": ids}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--git", default="git")
    args = parser.parse_args()
    build(args.checkout, args.output, args.git)
