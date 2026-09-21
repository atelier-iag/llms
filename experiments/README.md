# RoPE: first mechanism in milestone 3

Next mechanism: [GQA implementation, teaching example and comparison protocol](gqa.md).

Implementation: [rope.py](../reimplementation/rope.py), called from
[attention.py](../reimplementation/attention.py) after the Q/K projections and
before their scaled dot products. V is not rotated. The causal mask, softmax,
value mixing, residuals and MLP keep their previous behavior.

## Read and run the two-token example

From the repository root in the existing environment:

```sh
python -m experiments.rope_demo
```

The demo prints embeddings, positions, Q/K/V, rotated Q/K, masked scores, attention
weights and outputs with and without RoPE. It uses the same fictitious projections
as the teaching example: `Q = K = [a+b, 0]` and `V = [10a, 10b]` for `x = [a, b]`.
The actual two-dimensional RoPE rotates position 1 by **one radian (about 57.3°)**;
the earlier 90° rotation was illustrative, not the training recipe. The second
token's output is approximately `[4.1944, 5.8056]`, versus `[5, 5]` without RoPE.

## Rotation used by the model

Each head has 48 features, split into 24 adjacent pairs `(0,1), (2,3), ...`.
Pair `i` uses angular frequency `theta ** (-2*i / head_dim)`. Its angle at
position `p` is `p * frequency`, in radians. Rotate each pair `[a, b]` as:

```python
rotated_a = a * cos - b * sin
rotated_b = a * sin + b * cos
```

These are fixed rotations from the [RoFormer formulation](https://arxiv.org/abs/2104.09864),
not additional learned weights. Positions are generated as `0..length-1` for each
input window and are shared across heads and blocks. Generation still recomputes
the cropped context, with positions starting at zero again; shifting all positions
by the same offset preserves relative RoPE scores. There is no KV cache.

## Controlled comparison with the measured baseline

[rope_config.json](rope_config.json) differs from
[baseline_config.json](../reimplementation/baseline_config.json) only in the
experiment name and `model.rope_theta = 10000.0`. Omitting `rope_theta` (or setting
it to `null`) preserves the original model, including old checkpoints.

Both variants have **95,894,400 parameters**, identical initial learned weights
for the same seed, the same tokenizer, data order, optimizer, context, batch size
and **18,999,999 training targets / 9,279 updates** on the existing corpus.
The budget is matched in target tokens and updates; RoPE may change runtime.

Train the RoPE variant from scratch with the existing runner:

```sh
python -m reimplementation.train_baseline --device cuda --config experiments/rope_config.json
```

The runner creates a fresh ignored `runs/simple-baseline-*/` directory. Its
`config.json` and `metrics.json` identify `simple-384-rope`; checkpoints store the
RoPE setting in the model configuration. The final checkpoint reloads through the normal generation command.
Do not add RoPE to the already trained NoPE weights for this comparison.

**Status: full comparison completed on 2026-09-19.** Validation loss falls from
5.419455 without RoPE to 5.293822 with RoPE; perplexity falls from 225.756 to
199.103 (11.81%). Total recorded runtime rises from 38.44 to 55.44 minutes.
Greedy generations remain strongly repetitive. See the
[comparison report](../results/rope-baseline.md) and [metrics](../results/rope-baseline.json).
Local artifacts are in `runs/simple-baseline-r5x189so/`; the console log, launch
metadata and source snapshot are in `runs/rope-launch-ga4x7dhg/`.

Run the laboratory tests with `python -m pytest tests/ -q`.

Implementation checks performed: **70 tests passed**, including an independent
complex-number rotation reference, gradients, causality, relative-position
invariance and checkpoint round trips. The existing trained 96M NoPE checkpoint
also returned bit-identical probe logits before and after the change. One CUDA
update on the real 96M RoPE model with an `8 x 256` input batch completed on the
RTX 4060 Laptop GPU, with finite gradients and updated Q weights. This was only
an execution check, not the full comparison experiment.
