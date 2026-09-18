"""Use the same tokenizer JSON as the baseline, pinned for reproducibility."""

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


DOLMA_TOKENIZER = {
    "repo_id": "allenai/dolma2-tokenizer",
    "revision": "5292e5d6c0f40b67cc765fe41bec991cf4345b5c",
}


def load_tokenizer(spec: dict) -> Tokenizer:
    path = hf_hub_download(**spec, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer
