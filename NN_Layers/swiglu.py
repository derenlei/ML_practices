import torch
from torch.autograd import Function

class SwiGLUFunction(Function):
    """Custom SwiGLU implementation with forward and backward passes."""
    
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of SwiGLU.
        Args:
            ctx: Context object to save tensors for backward.
            x: Input tensor of shape (batch_size, 2 * feature_dim).
        Returns:
            Output tensor of shape (batch_size, feature_dim).
        """
        # Split input tensor into two parts
        x1, x2 = x.chunk(2, dim=-1)  # Divide along feature dimension
        
        # Compute Swish activation for x1: Swish(x) = x * sigmoid(x)
        swish_x1 = x1 * torch.sigmoid(x1)
        
        # Save for backward pass
        ctx.save_for_backward(x1, x2, swish_x1)
        
        # Return element-wise product of swish_x1 and x2
        return swish_x1 * x2

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of SwiGLU.
        Args:
            ctx: Context object to retrieve saved tensors.
            grad_output: Gradient of the loss with respect to the output.
        Returns:
            Gradient of the loss with respect to the input.
        """
        # Retrieve saved tensors
        x1, x2, swish_x1 = ctx.saved_tensors
        
        # Derivative of sigmoid(x)
        sigmoid_x1 = torch.sigmoid(x1)
        sigmoid_grad = sigmoid_x1 * (1 - sigmoid_x1)
        
        # Derivative of Swish(x): swish'(x) = sigmoid(x) + x * sigmoid'(x)
        swish_grad = sigmoid_x1 + x1 * sigmoid_grad
        
        # Gradients for x1 and x2
        grad_x1 = grad_output * (x2 * swish_grad)  # Chain rule
        grad_x2 = grad_output * swish_x1          # Chain rule
        
        # Combine gradients
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
        
        return grad_input
