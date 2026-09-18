# Building the model, one component at a time

The basic forward pass, loss and a first training loop are implemented:

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
`runs/reimplementation-tiny-batch-*/metrics.json` file. The model is trained
in memory; checkpoint saving is not implemented yet.

Next: a short run with separate validation data, checkpoint saving/loading, and
text generation. The tests cover gradient flow, gradient reset and learning the
fixed batch.

This is a teaching implementation using FP32 and basic causal attention. Explicit
positional encoding, Q/K normalization, GQA and mixed precision remain to be added
in later lessons. The components depend on PyTorch; some tests also compare with
the existing OLMo-core baseline.
