import torch
class Sigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        neg_mask = data < 0
        pos_mask = ~neg_mask

        zs = torch.empty_like(data)
        zs[neg_mask] = data[neg_mask].exp()
        zs[pos_mask] = (-data[pos_mask]).exp()

        res = torch.ones_like(data)
        res[neg_mask] = zs[neg_mask]

        result = res / (1 + zs)

        ctx.save_for_backward(result)
        return result
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (result, ) = ctx.saved_tensors
        grad = result * (1-result)
        return grad * grad_output
