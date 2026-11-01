"""Package portable source and evidence; no model weights, dependencies or secrets."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root.parent.parent / "Retrieval_Context_Evaluation_Project2.zip"
    excluded = {"__pycache__", ".git", ".venv", "models", "build", "_qa", "node_modules"}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (not path.is_file() or path.is_symlink()
                    or any(p in excluded or p.endswith(".egg-info") for p in relative.parts)
                    or path.suffix in {".pyc", ".partial"} or path.name == ".env"):
                continue
            archive.write(path, f"retrieval-eval-lab/{relative.as_posix()}")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "retrieval-eval-lab/reports/click-comparison.html" in archive.namelist()
        assert not any("model.onnx" in name for name in archive.namelist())
        print(f"{output}\n{len(archive.namelist())} files\n{output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
