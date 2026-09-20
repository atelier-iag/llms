"""Print the two-token attention example using the actual RoPE implementation."""

import torch

from reimplementation.attention import CausalAttentionHead


@torch.no_grad()
def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    x = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    positions = torch.arange(2)
    print('Tokens: "le", "chat"')
    print("Positions:", positions.tolist())
    print("Embeddings:\n", x[0])
    print("Real RoPE: with two features, position 1 rotates by 1 radian (57.3 degrees).")
    print("The earlier 90-degree rotation was only a teaching example.")

    for theta in (None, 10000.0):
        attention = CausalAttentionHead(2, 2, rope_theta=theta).double()
        attention.w_q.weight.copy_(torch.tensor([[1, 1], [0, 0]]))
        attention.w_k.weight.copy_(attention.w_q.weight)
        attention.w_v.weight.copy_(10 * torch.eye(2))
        q, k, v = attention.w_q(x), attention.w_k(x), attention.w_v(x)
        print("\nRoPE enabled:", theta is not None)
        print("Q before rotation:\n", q[0])
        print("K before rotation:\n", k[0])
        if attention.rope is not None:
            q, k = attention.rope(q, positions), attention.rope(k, positions)
        print("Q used for comparison:\n", q[0])
        print("K used for comparison:\n", k[0])
        print("V (no rotation):\n", v[0])
        scores = q @ k.transpose(-2, -1) / 2**0.5
        scores[:, 0, 1] = float("-inf")  # "le" cannot see the future "chat".
        print("Scaled scores with causal mask:\n", scores[0])
        output, weights = attention(x)
        print("Attention weights:\n", weights[0])
        print("Output = weights @ V:\n", output[0])


if __name__ == "__main__":
    main()
