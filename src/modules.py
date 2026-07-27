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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_dtype = input.dtype
        output = self.W_u(self.W_v(input.to(self.W_v.weight.dtype))) # input is casted to lowrank dtype
        return output.to(out_dtype) # output is casted back to model dtype
