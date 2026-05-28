import torch.nn as nn
import torch

class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_v = nn.Linear(in_features, rank, bias=False)
        self.W_u = nn.Linear(rank, out_features, bias=bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_dtype = input.dtype
        output = self.W_u(self.W_v(input.to(self.W_v.weight.dtype))) # input is casted to lowrank dtype
        return output.to(out_dtype) # output is casted back to model dtype