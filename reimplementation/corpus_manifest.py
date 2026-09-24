"""Check corpus identity before training, without opening the reserved holdout."""

import hashlib
import json
from pathlib import Path


def validate_manifest(data_dir, *, tokenizer, vocab_size, required=False):
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        if required:
            raise ValueError("this experiment requires a corpus manifest")
        return None  # Compatibility with the original baseline corpus.
    contents = path.read_bytes()
    manifest = json.loads(contents)
    if manifest.get("format_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("corpus manifest must describe a complete version-1 corpus")
    if (manifest.get("provenance", {}).get("tokenizer") != tokenizer
            or manifest.get("vocab_size") != vocab_size):
        raise ValueError("corpus tokenizer/vocabulary differs from the model")
    for split in ("train", "val"):
        entry = manifest.get("files", {}).get(split, {})
        if entry.get("path") != f"{split}.npy":
            raise ValueError(f"unexpected corpus filename for {split}")
        tokens = entry.get("tokens")
        file = Path(data_dir) / entry["path"]
        if not isinstance(tokens, int) or tokens < 1 or file.stat().st_size != tokens * 4:
            raise ValueError(f"corpus token count differs for {split}")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"corpus hash differs for {split}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest(),
            "provenance": manifest["provenance"]}
