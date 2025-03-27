import torch


class GELU(torch.autograd.Function):
    """The Gaussian Error Linear Units (GELU) activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        cdf = 0.5 * (1 + torch.erf(data / 2.0**0.5))
        ctx.save_for_backward(data, cdf)
        return data * cdf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        data, cdf = ctx.saved_tensors
        pdf_val = torch.distributions.Normal(0, 1).log_prob(data).exp()
        grad = cdf + data * pdf_val
        return grad_output * grad
