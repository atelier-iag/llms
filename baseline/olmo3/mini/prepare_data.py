from pathlib import Path
import io
import json
import os

import numpy as np
import requests
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_url
from transformers import AutoTokenizer


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_TOKENS = 19_000_000
VAL_TOKENS = 1_000_000

# Petit mix de domaines Dolma 3.
SOURCES = [
    "olmocr_science_pdfs-science_math_and_technology-part1",
    "olmocr_science_pdfs-education_and_jobs",
    "olmocr_science_pdfs-history_and_geography",
    "olmocr_science_pdfs-software_development",
]

REPO = "allenai/dolma3_pool"

tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
api = HfApi()

# Fichiers finaux utilisés ensuite par OLMo-core.
train = np.memmap(
    DATA_DIR / "train.npy",
    mode="w+",
    dtype=np.uint32,
    shape=(TRAIN_TOKENS,),
)

val = np.memmap(
    DATA_DIR / "val.npy",
    mode="w+",
    dtype=np.uint32,
    shape=(VAL_TOKENS,),
)

train_pos = 0
val_pos = 0

train_per_source = TRAIN_TOKENS // len(SOURCES)
val_per_source = VAL_TOKENS // len(SOURCES)

# Authentification optionnelle : fonctionne aussi sans HF_TOKEN.
headers = {}
if token := os.environ.get("HF_TOKEN"):
    headers["Authorization"] = f"Bearer {token}"


for source in SOURCES:
    print(f"\n=== {source} ===")

    source_train = 0
    source_val = 0

    # On récupère seulement la liste des shards de cette source.
    files = api.list_repo_tree(
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=f"data/{source}",
        recursive=False,
    )

    for file in files:
        if not file.path.endswith(".jsonl.zst"):
            continue

        print(f"Streaming {file.path}")

        url = hf_hub_url(
            repo_id=REPO,
            repo_type="dataset",
            filename=file.path,
        )

        # Le shard est lu directement à distance et décompressé en mémoire.
        with requests.get(
            url,
            stream=True,
            headers=headers,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()

            decompressor = zstd.ZstdDecompressor()

            with decompressor.stream_reader(response.raw) as reader:
                stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in stream:
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = document.get("text")
                    if not text:
                        continue

                    # Texte brut -> IDs du tokenizer OLMo.
                    ids = tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                    ids.append(tokenizer.eos_token_id)

                    # On remplit d'abord le quota train de cette source.
                    if source_train < train_per_source:
                        remaining = train_per_source - source_train
                        n = min(len(ids), remaining)

                        train[train_pos : train_pos + n] = ids[:n]

                        train_pos += n
                        source_train += n

                        # Ne pas mettre la fin du même document dans validation.
                        continue

                    # Puis les documents suivants alimentent validation.
                    if source_val < val_per_source:
                        remaining = val_per_source - source_val
                        n = min(len(ids), remaining)

                        val[val_pos : val_pos + n] = ids[:n]

                        val_pos += n
                        source_val += n

                    if (
                        source_train >= train_per_source
                        and source_val >= val_per_source
                    ):
                        break

        print(
            f"  train: {source_train:,}/{train_per_source:,} | "
            f"val: {source_val:,}/{val_per_source:,}"
        )

        # Inutile de télécharger d'autres shards une fois le quota atteint.
        if (
            source_train >= train_per_source
            and source_val >= val_per_source
        ):
            break


train.flush()
val.flush()

print("\nDone.")
print(f"Train: {train_pos:,} tokens")
print(f"Val:   {val_pos:,} tokens")