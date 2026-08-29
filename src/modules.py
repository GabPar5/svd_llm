import math

import torch
import torch.nn as nn

class LowRank(nn.Module):
    """
    Truncated-SVD replacement for an nn.Linear, computing W_u(W_v(x)).

    The factors may hold a different dtype than the rest of the model, so the
    input is cast on the way in and the output is cast back on the way out.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_v = nn.Linear(in_features, rank, bias=False)
        self.W_u = nn.Linear(rank, out_features, bias=bias)

    @property
    def factor_dtype(self) -> torch.dtype:
        return self.W_v.weight.dtype

    def dense_weight(self) -> torch.Tensor:
        """The (out, in) matrix this factorization stands for, for the checks"""
        return self.W_u.weight @ self.W_v.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_dtype = input.dtype
        output = self.W_u(self.W_v(input.to(self.W_v.weight.dtype))) # input is casted to lowrank dtype
        return output.to(out_dtype) # output is casted back to model dtype

class HeadBlockLowRank(nn.Module):
    """
    Head-partitioned truncated-SVD replacement for an nn.Linear.

    A projection whose output is the concatenation of independent attention
    heads is factored one head at a time, so the reconstruction is block
    diagonal in head space and a head's rank cannot be spent on another head's
    subspace. `W_v` stays one linear because every head reads the same input;
    only the second factor is per head.

    Under GQA this is what a joint factorization gets wrong. Each key or value
    head is read by `num_attention_heads / num_key_value_heads` query heads, so
    a single rank budget shared across the whole KV space lets one head's
    collapse propagate to every query head that reads it. It is also the
    cheaper parametrization: `heads * rank * (in + head_dim)` against the joint
    `rank_joint * (in + heads * head_dim)`, so at equal total rank the block
    form buys more of it.
    """

    def __init__(self, in_features: int, out_features: int, heads: int, rank: int, bias: bool):
        super().__init__()

        if out_features % heads != 0:
            raise ValueError(
                f"`out_features` {out_features} is not divisible by `heads` {heads}, "
                f"so the output does not partition into equal head blocks",
            )

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.head_dim = out_features // heads
        # Per head, unlike LowRank.rank which is the rank of the whole matrix
        self.rank = rank

        self.W_v = nn.Linear(in_features, heads * rank, bias=False)
        self.W_u = nn.Parameter(torch.empty(heads, self.head_dim, rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # `W_u` is a bare parameter rather than a linear, so nothing initializes
        # it on the way in. Matching nn.Linear keeps a module that is built but
        # not yet loaded usable instead of holding whatever the allocator left
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_u, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.rank) if self.rank > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def factor_dtype(self) -> torch.dtype:
        return self.W_v.weight.dtype

    def dense_weight(self) -> torch.Tensor:
        """The (out, in) matrix this factorization stands for, for the checks"""
        per_head = self.W_v.weight.unflatten(0, (self.heads, self.rank))
        return torch.einsum("hdr,hri->hdi", self.W_u, per_head).flatten(0, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_dtype = input.dtype
        projected = self.W_v(input.to(self.W_v.weight.dtype)) # input is casted to lowrank dtype
        per_head = projected.unflatten(-1, (self.heads, self.rank))
        output = torch.einsum("...hr,hdr->...hd", per_head, self.W_u).flatten(-2)

        if self.bias is not None:
            output = output + self.bias

        return output.to(out_dtype) # output is casted back to model dtype
