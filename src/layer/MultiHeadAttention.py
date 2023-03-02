import torch
from torch import nn
from .ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_h = d_model // h

        # number of heads, input dim, dim in each head
        self.W_K = nn.Parameter(torch.rand(h, d_model, self.d_h))
        self.W_Q = nn.Parameter(torch.rand(h, d_model, self.d_h))
        self.W_V = nn.Parameter(torch.rand(h, d_model, self.d_h))

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_h)
        self.linear = nn.Linear(h * self.d_h, d_model)

    def forward(
        self,
        Qs: torch.Tensor,
        Ks: torch.Tensor,
        Vs: torch.Tensor,
        mask_3d: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len = Qs.size(0), Qs.size(1)

        # head, batch_size, seq_len, d_model
        Qs = Qs.repeat(self.h, 1, 1, 1)
        Ks = Ks.repeat(self.h, 1, 1, 1)
        Vs = Vs.repeat(self.h, 1, 1, 1)

        Qs = torch.einsum("hbjk,hkl->hbjl", (Qs, self.W_Q))
        Ks = torch.einsum("hbjk,hkl->hbjl", (Ks, self.W_K))
        Vs = torch.einsum("hbjk,hkl->hbjl", (Vs, self.W_V))

        Qs = Qs.view(self.h * batch_size, seq_len, self.d_h)
        Ks = Ks.view(self.h * batch_size, seq_len, self.d_h)
        Vs = Vs.view(self.h * batch_size, seq_len, self.d_h)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        # (head*batch_size, seq_len, d_model)
        attention_output = self.scaled_dot_product_attention(Qs, Ks, Vs, mask_3d)

        attention_output = torch.chunk(attention_output, self.h, dim=0)
        attention_output = torch.cat(attention_output, dim=2)

        return self.linear(attention_output)
