# rms norm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# layer norm
class LayerNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # Fixed typo
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling factor
        self.bias = nn.Parameter(torch.zeros(dim))  # Learnable bias

    def forward(self, x):
        # Compute mean along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        # Compute variance along the last dimension
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # Normalize the input
        denominator = torch.sqrt(var + self.eps)
        normalized = (x - mean) / denominator
        # Apply learnable parameters
        output = normalized * self.weight + self.bias
        return output
    
      
# batch norm
class BatchNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # Fixed typo
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling factor
        self.bias = nn.Parameter(torch.zeros(dim))  # Learnable bias
    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0)
        output = (x-mean)/torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output
        
        
