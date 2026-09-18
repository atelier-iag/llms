# Building the model, one component at a time

The basic forward pass is implemented:

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

From the repository root, with the existing environment activated:

```sh
python -m reimplementation.model
python -m reimplementation.loss
```

The example uses six vocabulary tokens, four-number vectors, two attention heads,
eight feed-forward intermediate features, and two blocks. Its weights are random;
the printed scores are not trained predictions.

The loss example takes the toy excerpt `[2, 5, 2, 4]`, passes `[2, 5, 2]` to the
model, and scores its predictions against `[5, 2, 4]`. It uses unpadded batches
with a known target at every input position.

Next: study backpropagation, add the optimizer and implement the training loop.
The tests already check gradient flow through the full model using our loss.

This is a teaching implementation using FP32 and basic causal attention. Explicit
positional encoding, Q/K normalization, GQA and mixed precision remain to be added
in later lessons. The components depend on PyTorch; some tests also compare with
the existing OLMo-core baseline.
