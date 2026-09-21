# GQA: shared K/V projections and comparison with RoPE

The trained RoPE reference uses eight independent attention heads in
[attention.py](../reimplementation/attention.py). The GQA variant is implemented
in [gqa.py](../reimplementation/gqa.py) and selected by `num_kv_heads` in the
model configuration. Existing configurations and checkpoints keep using MHA.

Grouped Query Attention keeps distinct Q projections and shares K/V projections
within groups of heads. See the [GQA paper](https://arxiv.org/abs/2305.13245).
Our GQA variant uses eight query heads and two K/V groups:

| Query heads (zero-based) | Shared key projection | Shared value projection |
|---|---|---|
| 0, 1, 2, 3 | W_K of group 0 | W_V of group 0 |
| 4, 5, 6, 7 | W_K of group 1 | W_V of group 1 |

These are groups of **heads**, not groups of tokens. For every input token,
each group produces one 48-number K and one 48-number V. Each of the eight
heads still produces its own 48-number Q and its own attention weights over
the visible tokens. Sharing K/V does not force heads to attend identically.
Concatenation still yields 8 x 48 = 384 features per token before the output projection.

## Two-token teaching example

```sh
python -m experiments.gqa_demo
```

The demo studies only the second token, "chat", in "le chat". Every query shown
belongs to that same token; the two rows in K and V belong to the two visible
token positions. There are no future keys in this example. The vectors are
fictitious and two-dimensional; RoPE is omitted to isolate sharing.

For heads 0 and 1 in group 0:

```text
Shared K: [[1, 0], [0, 1]]
Shared V: [[10, 0], [0, 10]]

Head 0: Q = [1, 0] -> weights [0.6698, 0.3302] -> output [6.6976, 3.3024]
Head 1: Q = [0, 1] -> weights [0.3302, 0.6698] -> output [3.3024, 6.6976]
```

The same K/V serve different queries, so the outputs differ. The core loop in
[gqa_demo.py](gqa_demo.py) selects the group, then keeps the usual attention calculation:

```python
group_id = head_id // heads_per_group
k = keys[group_id]
v = values[group_id]
scores = (q @ k.T) / math.sqrt(head_dim)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

## Model integration

[gqa.py](../reimplementation/gqa.py) computes K and V once per group. RoPE rotates
each group's K once and each head's Q separately; V remains unrotated. Query head
`h` selects group `h // heads_per_group`, applies the same scaled dot product,
causal mask and softmax as MHA, then mixes that group's V. Gradients from the
query heads accumulate in their shared K/V projections.

`CausalLanguageModel(..., num_kv_heads=2, rope_theta=10000.0)` passes the option
through the decoder to every block. `None` (the default) keeps the original MHA
implementation and state-dict keys. A positive `num_kv_heads` must divide
`num_heads`; one K/V group is MQA, and one group per query head is equivalent to
MHA when the weights match. Checkpoints record the selected architecture and
reload through the existing generation command.

## Measured comparison protocol

Going from eight K/V heads to two divides the number of stored K/V vectors by
four. A future KV cache would therefore use one quarter of the K/V storage at
the same context length and dtype. Our current generator has **no KV cache**;
this ratio is not a measured fourfold reduction in total memory or runtime.

The Q/K dot products and weighted value sums still run for all eight query heads.
The model also has fewer K/V projection parameters: **94,124,928** total
parameters instead of **95,894,400**, with embedding/output tables unchanged.
Keeping the data and training recipe fixed will test the quality/cost tradeoff;
equal parameter count is no longer part of this comparison.

[gqa_config.json](gqa_config.json) differs from [rope_config.json](rope_config.json)
only in its name and `model.num_kv_heads = 2`. RoPE stays enabled. The comparison
uses the same tokenizer, data files, seed 0, shuffled windows, context 256, batch
8, FP32, optimizer, learning-rate schedule, evaluations and prompts. One corpus
pass gives **18,999,999 targets and 9,279 updates**. The final validation evaluates
all 999,999 targets. Training starts from random weights, not the trained RoPE
checkpoint; conversion/uptraining from the paper is a separate experiment.

The same seed does not imply identical initial weights between these different
parameter layouts. This first run measures the quality/cost tradeoff for one
initialization; it does not establish an effect across seeds.

```sh
python -m reimplementation.train_baseline --device cuda --config experiments/gqa_config.json
```

The runner creates a fresh `runs/simple-baseline-*/` directory. Its configuration
identifies `simple-384-rope-gqa2`; final metrics and the inference checkpoint are
saved there. Compare with the [completed RoPE reference](../results/rope-baseline.md).

**Status: integrated, 96 laboratory tests passing; first full training launched
on 2026-09-21.** Local run: `runs/simple-baseline-se1nrgls/`; console log, launch
metadata and source snapshot: `runs/gqa-launch-fweigtsy/`. Training code commit:
`3114f0c0c78febc3ffed318850e0fb64187938af`. The `complete` log event and final
`metrics.json` establish completion; this note only records the launch.

Tests compare outputs and all parameter gradients with PyTorch
SDPA, verify that each K/V projection runs once, enforce causality, check the MHA
limit, count real-model parameters, and exercise training/evaluation/checkpoint
reload/generation on synthetic data. The saved NoPE and RoPE reference checkpoints
also reproduce bit-identical probe logits after this integration.

A real CUDA update was also checked on the full 94,124,928-parameter variant,
using eight 256-token sequences from the existing corpus: the loss and every
gradient were finite, and the shared K projection weights were updated. This
execution check does not measure trained quality or steady-state throughput.
