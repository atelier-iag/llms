# Building the model, one component at a time

The basic model and a complete training, validation and generation pipeline are implemented:

```text
Token IDs -> embeddings -> Transformer blocks -> final RMSNorm -> vocabulary scores
```

Read the components in this order:

1. [embeddings.py](embeddings.py): look up a learned vector for each token.
2. [attention.py](attention.py): causal attention heads, concatenation and output projection.
   Optional [rope.py](rope.py) rotates Q/K pairs using token positions (milestone 3).
   Optional [gqa.py](gqa.py) shares K/V projections between groups of query heads.
3. [normalization.py](normalization.py): RMSNorm.
4. [feed_forward.py](feed_forward.py): the SwiGLU feed-forward network.
5. [block.py](block.py): attention and feed-forward updates, each normalized before residual addition.
6. [decoder.py](decoder.py): independently learned blocks applied in sequence.
7. [lm_head.py](lm_head.py): final normalization and one score per vocabulary token.
8. [model.py](model.py): connect all components.
9. [loss.py](loss.py): shift inputs and targets, then compute mean cross-entropy.
10. [train.py](train.py): clear gradients, compute the loss, backpropagate and update weights.
11. [data.py](data.py): read real token windows from the baseline's existing files.
12. [train_corpus.py](train_corpus.py): train, evaluate, save and verify a checkpoint.
13. [generate.py](generate.py): reload the model and generate one new token at a time.

From the repository root, with the existing environment activated:

```sh
python -m reimplementation.model
python -m reimplementation.loss
```

These examples use six vocabulary tokens, four-number vectors, two attention heads,
eight feed-forward intermediate features, and two blocks. The model and loss examples
start from random weights; their printed scores are not trained predictions.

The loss example takes the toy excerpt `[2, 5, 2, 4]`, passes `[2, 5, 2]` to the
model, and scores its predictions against `[5, 2, 4]`. It uses unpadded batches
with a known target at every input position.

## First training check

Train the same 388-parameter model repeatedly on the fixed batch `[2, 5, 2, 4]`:

```sh
python -m reimplementation.train --steps 200 --lr 0.01 --seed 0
```

This runs on one CPU thread with AdamW, betas `(0.9, 0.95)` and no weight decay.
Each update resets gradients, calculates the loss, calls `backward()`, then calls
`optimizer.step()`. Progress logs show the loss before the numbered update; the
final loss is measured again after the last update.

The first run reduced loss from **1.740844 to 0.000521** and learned all three
targets `[5, 2, 4]`. This checks that the model can memorize a tiny batch; it does
not measure generalization. See the [experiment summary](../results/reimplementation-tiny-batch.md).

Each run writes its configuration, loss history and final predictions to a new
`runs/reimplementation-tiny-batch-*/metrics.json` file. This small teaching example
trains in memory; the corpus command below also saves the model weights.

## Complete pipeline on real tokens

Use the existing environment (Python 3.11+, PyTorch, NumPy, `tokenizers` and
`huggingface_hub`) and the already prepared `train.npy` / `val.npy` files:

```sh
python -m reimplementation.train_corpus --device cuda --steps 200 --seed 0
```

This is a single Python process on one GPU. `--device cpu` is also supported.
The default model has 64 features, four heads, two blocks, 256 feed-forward
features and 100,278 vocabulary entries: **12,966,976 parameters**. It starts
from random weights; no baseline model weights are loaded. It reuses our existing
`train_step`, with AdamW, learning rate 0.001, betas `(0.9, 0.95)` and weight decay 0.01.

The data files are **raw uint32 arrays despite their `.npy` suffix**. The loader
opens them read-only, samples random windows across the training file, and copies
each window into a PyTorch integer tensor. A window contains 129 tokens:

```python
inputs = tokens[:, :-1]   # first 128 tokens
targets = tokens[:, 1:]   # next token for each of the 128 input positions
```

Each update processes two windows, so 200 updates score 51,200 training targets.
Windows may cross document boundaries separated by EOS; attention is causal but
does not reset at EOS. There is no padding. This command reads existing data;
it does not run `prepare_data.py` or replace either source file.

Before and after training, [evaluation/language_model.py](../evaluation/language_model.py)
scores the same 32 fixed windows from each split, spread across the files
(4,096 targets per split). Validation uses `val.npy` and never updates weights.
These are small validation samples, **not a final holdout** or a full-corpus evaluation.
Loss is the mean negative log probability per target; perplexity is `exp(loss)`.

Each invocation creates a new ignored `runs/reimplementation-corpus-*/` directory:

- `metrics.json`: configuration, data hashes, evaluation offsets, before/after
  metrics, training loss history, timing, checkpoint check and generated text.
- `model.pt`: architecture, learned weights, completed update count, context
  length and tokenizer reference. It is an inference checkpoint; optimizer state
  and training resume are not implemented.

The tokenizer JSON is pinned to the revision in [tokenizer.py](tokenizer.py),
loaded from the Hugging Face cache or downloaded if absent. The checkpoint stores
this reference so text is decoded with the same vocabulary. The model is reloaded
and its scores checked against the saved model before generating the example.

To generate again, replace the checkpoint path with the one printed by training:

```sh
python -m reimplementation.generate runs/reimplementation-corpus-REPLACE/model.pt \
  --device cuda --prompt "The purpose of science is" --max-new-tokens 32
```

At each iteration, generation feeds the latest context (up to 128 tokens) through
the model, selects the highest-scoring next token from the **last position**, and
appends it. It stops at EOS or the requested limit. This version recomputes the
context and uses greedy decoding; it has no KV cache.

The first real-data run reduced validation loss from **11.7044 to 7.7851** and
reloaded the checkpoint with zero logit difference. Generated text is still
repetitive after this short run. See the [measured results](../results/reimplementation-corpus.md)
and [recorded metrics](../results/reimplementation-corpus.json).

Run all laboratory checks with `python -m pytest tests/ -q`. The pipeline tests
use tiny synthetic data and require neither a GPU nor a tokenizer download.

## Simple model baseline before milestone 3

The short 64-feature run above is a pipeline check. For the reference experiment,
[train_baseline.py](train_baseline.py) reads [baseline_config.json](baseline_config.json):
384 features, eight blocks, eight heads, a 1,536-feature MLP, context length 256,
batch size eight, and **95,894,400 parameters**. The embedding table has shape
`[100278, 384]`. This baseline configuration has no explicit positional encoding, Q/K
normalization, GQA or mixed precision. It starts from random weights.

```sh
python -m reimplementation.train_baseline --device cuda
```

The script creates a new `runs/simple-baseline-*/` directory. It reads the existing
data files without modifying them and visits every next-token target **once**:

- divide the training file into consecutive windows of 256 targets;
- shuffle those windows with seed zero to mix the corpus domains;
- neighboring windows share one context token, but no target;
- process any incomplete last batch and short final window without padding.

For the existing 19-million-token file this gives **18,999,999 targets in 9,279
updates**. Only the very first token has no preceding context and cannot be a
target. The last window has 191 targets. Context does not carry between windows.
Windows can cross document boundaries; the causal mask does not reset at EOS.

The fixed recipe uses AdamW (peak learning rate `3e-4`, betas `(0.9, 0.95)`, weight
decay `0.1` on all parameters), 100 warmup updates, cosine decay to `3e-5`, and
gradient norm clipping at 1.0. The forward pass, loss and basic attention remain
the same as the teaching implementation. The optional clipping argument in
`train_step` leaves the earlier tiny-batch and short-corpus commands unchanged.

Evaluation never updates weights:

- before and after training: full validation, **999,999 targets**, including the
  short final window;
- at update zero, every 1,000 updates and at the end: the same 128 fixed windows
  from each split, **32,768 targets per split**;
- training logs every 100 updates report the token-weighted mean of losses
  measured **before each update**, while weights change. These differ from the
  fixed-model evaluations of the fixed train/validation samples.

Run artifacts:

- `config.json`: the exact recipe used;
- `progress.jsonl`: incremental training and evaluation measurements;
- `recovery.pt`: latest periodic model, optimizer, update count, data hashes and
  Torch RNG states, written atomically, with histories and partial logging windows;
- `model.pt`: final inference checkpoint compatible with `reimplementation.generate`;
- `metrics.json`: full results, evaluation offsets, data hashes, timing and three
  greedy generations from the reloaded final checkpoint.

Resume an interrupted baseline, RoPE or GQA run from its last saved update:

```sh
python -m reimplementation.train_baseline --device cuda \
  --resume runs/simple-baseline-REPLACE/recovery.pt
```

The checkpoint supplies the configuration; an explicit `--config` must match it.
The runner verifies the tokenizer, data hashes and target count, restores the
weights, AdamW state and Torch RNG states, then regenerates the seeded epoch
permutations and skips the already completed batches. The learning-rate schedule
continues at the next update. Use the same CPU/CUDA device configuration.
A **new run directory** keeps the source checkpoint and its logs intact. Updates
after the last saved checkpoint are repeated, never counted twice in the result.

Older `recovery.pt` files need the adjacent `progress.jsonl` to recover the initial
measurements and history. They are supported when the saved update coincides with
a completed training-log boundary (as in the 1,000-update checkpoints of our runs).
New checkpoints also preserve a partially filled logging window and need no old log.
`model.pt` is for inference and cannot restore the optimizer.

For a resumed run, `metrics.json` retains the complete evaluation/training histories
and total token count, but reports timing, throughput and GPU peak memory for the
new process under `session`. The corresponding top-level full-run cost fields
are `null`: older checkpoints lack the measurements needed to reconstruct those
totals reliably. Each new training-log record identifies its `session_start_step`;
its elapsed time is measured within that process. Do not compare a resumed
session's time with the full NoPE or RoPE training times.

The final checkpoint is checked for identical logits after reconstruction. The
generation command is the same as above, with the new run's `model.pt` path.
The [baseline report](../results/simple-baseline.md) records the measurements and
limitations. Keep this recipe, tokenizer, data order, splits and token budget
fixed when comparing the [RoPE variant](../experiments/README.md); train each variant from scratch.

The completed run processed all **18,999,999 training targets** and reduced full
validation loss from **11.677568 to 5.419455** (perplexity **225.756**). It took
36.09 minutes for the training loop and 38.44 minutes including the recorded
evaluations, saves and generations. Greedy text remains strongly repetitive.
The checkpoint reproduced identical logits and the first generation was also
verified in a fresh Python process. See the [recorded metrics](../results/simple-baseline.json).

Rebuild the published plots with Matplotlib:

```sh
python -m evaluation.plot_baseline results/simple-baseline.json \
  --output-prefix results/simple-baseline-curves
```

This is a teaching implementation using FP32 and basic causal attention. RoPE is
now available via `rope_theta=10000.0`; the default `None` keeps the original NoPE
baseline and its saved checkpoints compatible. See the [executable two-token demo
and controlled comparison recipe](../experiments/README.md). The completed RoPE
run reaches full-validation perplexity **199.103**, versus **225.756** without
RoPE, with repetition still present; see the [comparison](../results/rope-baseline.md).
GQA is available via `num_kv_heads=2`, keeping eight independent queries and two
K/V groups per block; see the [implementation and comparison recipe](../experiments/gqa.md).
Omitting that option preserves MHA and old checkpoints. Optional training precision
`bf16` now enables autocast while retaining FP32 weights, optimizer state and evaluation.
See the [mixed precision protocol](../experiments/mixed_precision.md) and
[BF16 configuration](../experiments/bf16_config.json). The default remains FP32;
recovery checkpoints preserve and validate the training precision.
Optional `attention_backend="sdpa"` packs the existing head projections and uses
PyTorch causal SDPA, including FlashAttention for the supported BF16 CUDA shapes.
The default remains manual attention. See the [SDPA protocol](../experiments/sdpa.md).

For milestone 4, [prepare_corpus.py](prepare_corpus.py) builds a new versioned
train/dev/holdout corpus with pinned sources/tokenizer and document-level splits.
It additionally requires `requests` and `zstandard`. Follow the
[50M pretraining protocol](../experiments/pretraining_50m.md) to prepare and train.
The runner checks a present corpus manifest and requires it for that configuration;
incomplete or altered training data are rejected before creating a run. The
reserved holdout is not read by the training runner.
For a controlled data-budget comparison, [subset_corpus.py](subset_corpus.py)
extracts a smaller balanced training set from a prepared corpus and copies dev
and holdout byte-for-byte. See the [25M/50M protocol](../experiments/data_scaling.md).
Q/K normalization remains later work. The components depend on PyTorch; some
tests also compare with the existing OLMo-core baseline.
