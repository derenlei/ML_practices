import torch
# forward only
def relu(z: float) -> float:
	return max(0, z)

def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
  # return z if z > 0 else alpha * z
	return max(alpha*z, z)

# forward and backward
class ReLU(torch.autograd.Function):
    """The Rectified Linear Unit (ReLU) activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        ctx.save_for_backward(data)
        return torch.where(data < 0.0, 0.0, data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        (data,) = ctx.saved_tensors
        grad = torch.where(data <= 0, 0, 1)
        return grad_output * grad


class LeakyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha: float = 1e-2) -> torch.Tensor:
        ctx.save_for_backward(data, torch.tensor(alpha).double())
        return torch.where(data < 0.0, data*alpha, data)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        data, alpha = ctx.saved_tensors
        grad = torch.where(data < 0.0, alpha, 1)
        return grad_output * grad
      
