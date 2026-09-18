"""Read the baseline's raw uint32 token files without modifying them."""

from pathlib import Path

import numpy as np
import torch


class TokenFile:
    """Despite their .npy suffix, these files have no NumPy header.

    Each example contains sequence_length + 1 consecutive tokens: the extra
    token is the last target. Windows can cross an EOS document boundary.
    """

    def __init__(self, path: Path, sequence_length: int, vocab_size: int):
        self.path = Path(path).resolve()
        self.sequence_length = sequence_length
        if sequence_length < 1 or vocab_size < 1:
            raise ValueError("sequence_length and vocab_size must be positive")
        size = self.path.stat().st_size
        if size % 4 or size < 4 * (sequence_length + 1):
            raise ValueError("expected a raw uint32 file with at least one full window")
        with self.path.open("rb") as stream:
            if stream.read(6) == b"\x93NUMPY":
                raise ValueError("expected raw uint32 tokens, not a NumPy .npy container")
        self.tokens = np.memmap(self.path, mode="r", dtype=np.uint32)
        if int(self.tokens.max()) >= vocab_size:
            raise ValueError("token file contains an ID outside the model vocabulary")

    @property
    def num_starts(self) -> int:
        return len(self.tokens) - self.sequence_length

    def batch(self, starts: np.ndarray) -> torch.Tensor:
        starts = np.asarray(starts)
        if starts.ndim != 1 or not len(starts) or starts.dtype.kind not in "iu":
            raise ValueError("starts must be a nonempty vector of integer offsets")
        if np.any(starts < 0) or np.any(starts >= self.num_starts):
            raise ValueError("window offset outside the token file")
        # Copy into writable int64 storage for PyTorch; the source stays read-only.
        windows = np.stack([
            self.tokens[int(start):int(start) + self.sequence_length + 1]
            for start in starts
        ]).astype(np.int64)
        return torch.from_numpy(windows)

    def sample(self, rng: np.random.Generator, batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self.batch(rng.integers(self.num_starts, size=batch_size))

    def evaluation_starts(self, num_windows: int) -> np.ndarray:
        """Fixed offsets spread across the file, covering the baseline's domains."""
        if not 1 <= num_windows <= self.num_starts:
            raise ValueError("num_windows must fit within the available start offsets")
        return np.linspace(0, self.num_starts - 1, num_windows, dtype=np.int64)

    def epoch_batch_count(self, batch_size: int) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        full_windows, remainder = divmod(len(self.tokens) - 1, self.sequence_length)
        return (full_windows + batch_size - 1) // batch_size + bool(remainder)

    def epoch_batches(self, batch_size: int, rng: np.random.Generator | None = None):
        """Score every token except the first exactly once, with no padding.

        Adjacent windows share one boundary token as context, but never a target.
        Shuffle full windows for training; keep file order when rng is None.
        A short final window handles the remainder without discarding data.
        """
        self.epoch_batch_count(batch_size)  # Validate before iterating.
        full_windows, remainder = divmod(len(self.tokens) - 1, self.sequence_length)
        starts = np.arange(full_windows, dtype=np.int64) * self.sequence_length
        if rng is not None:
            rng.shuffle(starts)
        for index in range(0, len(starts), batch_size):
            yield self.batch(starts[index:index + batch_size])
        if remainder:
            tail = np.array(self.tokens[full_windows * self.sequence_length:], dtype=np.int64)
            yield torch.from_numpy(tail).unsqueeze(0)


def open_splits(data_dir: Path, sequence_length: int, vocab_size: int):
    train_path = Path(data_dir) / "train.npy"
    val_path = Path(data_dir) / "val.npy"
    if train_path.samefile(val_path):
        raise ValueError("training and validation must use separate files")
    return (
        TokenFile(train_path, sequence_length, vocab_size),
        TokenFile(val_path, sequence_length, vocab_size),
    )
