# Building the model, one component at a time

The basic model and a complete training, validation and generation pipeline are implemented:

```text
Token IDs -> embeddings -> Transformer blocks -> final RMSNorm -> vocabulary scores
```

Read the components in this order:

1. [embeddings.py](embeddings.py): look up a learned vector for each token.
2. [attention.py](attention.py): causal attention heads, concatenation and output projection.
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

This is a teaching implementation using FP32 and basic causal attention. Explicit
positional encoding, Q/K normalization, GQA and mixed precision remain to be added
in later milestones. The components depend on PyTorch; some tests also compare with
the existing OLMo-core baseline.
