"""Pinned CPU sentence embeddings. No source code from the model repo is loaded."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import urllib.request

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model(directory):
    directory = Path(directory)
    for name, expected in FILES.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or incorrect model file: {name}. Run retrieval_lab.download_model.")
    return {"model_id": MODEL_ID, "revision": REVISION, "files_sha256": dict(FILES)}


def download_model(directory):
    """Explicit network operation; inference itself never downloads anything."""
    directory = Path(directory)
    for name, expected in FILES.items():
        path = directory / name
        if path.is_file() and sha256(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha256(partial) != expected:
                raise ValueError(f"Download checksum mismatch: {name}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return verify_model(directory)


def mean_pool_normalize(hidden, mask):
    """Attention-masked mean pooling and L2 normalization, per the model card."""
    import numpy as np
    hidden, mask = np.asarray(hidden), np.asarray(mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not np.isfinite(hidden).all():
        raise ValueError("invalid embedding output shape or nonfinite values")
    if not np.isin(mask, [0, 1]).all() or (mask.sum(axis=1) == 0).any():
        raise ValueError("attention masks must be binary and nonempty")
    pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("invalid or zero embedding vector")
    return (pooled / norms).astype(np.float32)


class WordPieceCounter:
    """Full-context WordPiece count; excludes model special tokens and prompts."""
    name = "MiniLM WordPiece (no special tokens; not a chat-model tokenizer)"

    def __init__(self, directory):
        from tokenizers import Tokenizer
        self.provenance = verify_model(directory)
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prefix(self, text, budget):
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= budget:
            return text
        if budget == 0:
            return ""
        # Preserve original case/Unicode with source offsets, rather than decoding
        # normalized WordPiece IDs. Recount the boundary to guarantee the limit.
        end = encoding.offsets[budget][0]
        while end and self.count(text[:end]) > budget:
            end -= 1
        return text[:end]


class MiniLMEncoder:
    """CPU ONNX inference with 256-token input limit, batching, and truncation audit."""
    def __init__(self, directory):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        self.provenance = verify_model(directory)
        self.provenance.update({"onnxruntime": version("onnxruntime"),
                                "tokenizers": version("tokenizers"), "numpy": version("numpy"),
                                "provider": "CPUExecutionProvider", "dimensions": 384,
                                "max_sequence_tokens": 256, "pooling": "attention_mask_mean_then_l2"})
        self.full_tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.full_tokenizer.no_truncation()
        self.full_tokenizer.no_padding()
        self.tokenizer = Tokenizer.from_str(self.full_tokenizer.to_str())
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(Path(directory) / "onnx/model.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        if not self.input_names <= {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unexpected ONNX model input names")

    def encode(self, texts, batch_size=16):
        import numpy as np
        if type(batch_size) is not int or batch_size < 1 or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError("nonempty text list and positive batch size required")
        vectors, truncated = [], []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            truncated.extend(start+i for i, t in enumerate(batch)
                             if len(self.full_tokenizer.encode(t).ids) > 256)
            encodings = self.tokenizer.encode_batch(batch)
            inputs = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                      "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                      "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64)}
            output = self.session.run(None, {name: inputs[name] for name in self.input_names})[0]
            pooled = mean_pool_normalize(output, inputs["attention_mask"])
            if pooled.shape != (len(batch), 384):
                raise ValueError("unexpected embedding dimensions")
            vectors.append(pooled)
        return np.concatenate(vectors), truncated
