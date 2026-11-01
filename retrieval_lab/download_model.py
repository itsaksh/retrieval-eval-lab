import argparse
import json
from pathlib import Path
from .model import download_model


def main():
    parser = argparse.ArgumentParser(description="Download and SHA-256 verify the pinned public MiniLM ONNX model (91 MB)")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(download_model(args.directory), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
