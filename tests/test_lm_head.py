import torch

from reimplementation.lm_head import LMHead


def test_each_output_coordinate_scores_its_vocabulary_token():
    head = LMHead(d_model=4, vocab_size=6)
    with torch.no_grad():
        head.w_out.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    # RMSNorm turns [3, 4, 0, 0] into approximately [1.2, 1.6, 0, 0].
    logits = head(torch.tensor([[[3.0, 4.0, 0.0, 0.0]]]))
    expected = torch.tensor([[[1.2, 1.6, -1.2, 2.8, 0.0, 0.0]]])
    torch.testing.assert_close(logits, expected)
