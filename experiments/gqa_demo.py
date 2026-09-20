"""GQA example: one querying token, eight query heads, two shared K/V groups.

All vectors are invented. This isolates K/V sharing; it is not a trained model
or a replacement for the current multi-head attention implementation.
"""

import math

import torch


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    # Eight queries for the SAME token (position 1), one per attention head.
    queries = torch.tensor([
        [1., 0.], [0., 1.], [-1., 0.], [0., -1.],
        [1., 1.], [-1., 1.], [1., -1.], [-1., -1.],
    ])
    # Two groups; each holds one K and one V for each of two visible tokens.
    # Rows inside each group correspond to token positions 0 and 1.
    keys = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [1., -1.]]])
    values = torch.tensor([[[10., 0.], [0., 10.]], [[20., 0.], [0., 20.]]])
    heads_per_group = len(queries) // len(keys)
    head_dim = queries.shape[-1]

    print('Querying token: "chat" at position 1; visible tokens: "le", "chat".')
    print("8 query heads, 2 K/V groups, 2 features per vector (48 in our real model).")
    print("Heads 0-3 share group 0; heads 4-7 share group 1.")
    for group_id in range(len(keys)):
        print(f"\nGroup {group_id}: K = {keys[group_id].tolist()}")
        print(f"Group {group_id}: V = {values[group_id].tolist()}")

    outputs = []
    for head_id, q in enumerate(queries):
        group_id = head_id // heads_per_group
        k = keys[group_id]
        v = values[group_id]
        scores = (q @ k.T) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        outputs.append(output)
        print(f"\nHead {head_id}, group {group_id}, Q = {q.tolist()}")
        print("  weights on [le, chat]:", weights)
        print("  output:", output)

    print("\nConcatenated outputs for this token (before output projection):")
    print(torch.cat(outputs))


if __name__ == "__main__":
    main()
