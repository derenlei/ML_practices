import torch
from torch.autograd import Function

class GLUFunction(Function):
    """Custom GLU implementation with forward and backward passes."""
    
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of GLU.
        Args:
            ctx: Context object to save tensors for backward pass.
            x: Input tensor of shape (batch_size, 2 * feature_dim).
        Returns:
            Output tensor of shape (batch_size, feature_dim).
        """
        # Split the input tensor into two parts
        x1, x2 = x.chunk(2, dim=-1)  # Split along the feature dimension
        
        # Apply the sigmoid activation to x2
        gate = torch.sigmoid(x2)
        
        # Compute the GLU output: x1 * sigmoid(x2)
        output = x1 * gate
        
        # Save tensors for backward computation
        ctx.save_for_backward(x1, gate)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of GLU.
        Args:
            ctx: Context object to retrieve saved tensors.
            grad_output: Gradient of the loss with respect to the output.
        Returns:
            Gradient of the loss with respect to the input.
        """
        # Retrieve saved tensors
        x1, gate = ctx.saved_tensors
        
        # Gradients for x1 and x2
        grad_x1 = grad_output * gate  # Derivative of x1 * sigmoid(x2) w.r.t. x1
        grad_gate = grad_output * x1  # Derivative w.r.t. sigmoid(x2)
        
        # Derivative of sigmoid(x2)
        grad_x2 = grad_gate * gate * (1 - gate)  # chain rule
        
        # Concatenate gradients for x1 and x2
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
        
        return grad_input
