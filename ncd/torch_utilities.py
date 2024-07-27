import torch
from torch import nn
import math
from typing import Any

class Multilinear(nn.Module):
    """ A learned linear linear which supports tuple axis sizes. """
    def __init__(self, in_size  : tuple[int] | int, out_size : tuple[int] | int, 
        bias : bool = True, device : Any | None = None, dtype : Any | None = None) -> None:
        super().__init__()
        # Set the parameters
        def get_size(x):
            return (x, math.prod(x)) if isinstance(x, tuple) else ((x,), x)
        self.in_size,  self.in_features  = get_size(in_size)
        self.out_size, self.out_features = get_size(out_size)
        # Set up the linear module
        self.linear = nn.Linear(self.in_features, self.out_features, bias, device, dtype)
    
    def forward(self, x_in : torch.Tensor):
        # Reshape the input. The last axes should match, else there's an error.
        x_in = x_in.reshape(
            x_in.shape[:-len(self.in_size)] + (self.in_features,))
        # Apply the linear over the last axis.
        x = self.linear(x_in)
        # Return the proper output.
        return x.reshape(
            x.shape[:-1] + self.out_size)