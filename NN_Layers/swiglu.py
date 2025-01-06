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
        
# --------------------------LLAMA3-version-with-weights-------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_weight, w1_bias, w2_weight, w2_bias):
        """
        Forward pass for Swiglu
        Arguments:
        - ctx: Context to save tensors for backward computation.
        - x: Input tensor.
        - w1_weight, w1_bias: Weights and biases for w1.
        - w2_weight, w2_bias: Weights and biases for w2.
        """
        # Linear operations
        z = F.linear(x, w1_weight, w1_bias)
        silu = z * torch.sigmoid(z)
        w2 = F.linear(x, w2_weight, w2_bias)
        output = w2 * silu

        # Save for backward
        ctx.save_for_backward(x, z, silu, w1_weight, w2_weight)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, z, silu, w1_weight, w2_weight, w2_bias = ctx.saved_tensors
    
        # Compute gradients
        sigmoid_z = torch.sigmoid(z)
        grad_silu = sigmoid_z + z * sigmoid_z * (1 - sigmoid_z)  # Derivative of SiLU
        w2 = F.linear(x, w2_weight, w2_bias)
        grad_z = grad_output * w2 * grad_silu 
    
        grad_w1_weight = grad_z.T @ x
        grad_w1_bias = grad_z.T.sum(dim=0)
        grad_w2_weight = (grad_output  * silu).T @ x
        grad_w2_bias = (grad_output * silu).sum(dim=0)

        grad_x = (grad_output * silu) @ w2_weight +  grad_z @ w1_weight
        return  grad_x

        

# Example usage
class SwigluModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=True)
        self.w2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        return Swiglu.apply(x, self.w1.weight, self.w1.bias, self.w2.weight, self.w2.bias)
