# GQA: understand K/V sharing before changing the model

The trained RoPE model still uses eight independent attention heads. Each head
has its own learned Q, K and V projections in
[attention.py](../reimplementation/attention.py).

Grouped Query Attention keeps distinct Q projections and shares K/V projections
within groups of heads. See the [GQA paper](https://arxiv.org/abs/2305.13245).
Our proposed next variant uses eight query heads and two K/V groups:

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

## What the next model change will measure

Going from eight K/V heads to two divides the number of stored K/V vectors by
four. A future KV cache would therefore use one quarter of the K/V storage at
the same context length and dtype. Our current generator has **no KV cache**;
this ratio is not a measured fourfold reduction in total memory or runtime.

The Q/K dot products and weighted value sums still run for all eight query heads.
The model would also have fewer K/V projection parameters: **94,124,928** total
parameters instead of **95,894,400**, with embedding/output tables unchanged.
Keeping the data and training recipe fixed will test the quality/cost tradeoff;
equal parameter count is no longer part of this comparison.

**Status: executable teaching example only.** Integration into the model,
shared-gradient/causality/checkpoint tests and a measured training comparison
are the next implementation step. The existing trained model remains MHA + RoPE.
