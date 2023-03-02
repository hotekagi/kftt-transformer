import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Q, K, V: size of (N, D)
    where
        N: num of words
        D: dim of a word embedding
    """

    def __init__(self, D: int) -> None:
        super().__init__()
        self.D = D

    # Attention(Q,K,V) = softmax(Q・K^T / √D)・V
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attention_weight: torch.Tensor = torch.matmul(Q, torch.transpose(K, 1, 2)) / np.sqrt(self.D)

        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(mask, -torch.finfo(torch.float).max)

        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, V)
