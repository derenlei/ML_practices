import torch

class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the softmax function.
        Arguments:
        - ctx: Context to save tensors for backward computation.
        - x: Input tensor.
        Returns:
        - softmax: Softmax of the input tensor.
        """
        # Shift the input for numerical stability
        max_vals = x.max(dim=-1, keepdim=True).values
        shifted_x = x - max_vals
        
        # Compute softmax
        exp_x = torch.exp(shifted_x)
        softmax = exp_x / exp_x.sum(dim=-1, keepdim=True)
        
        # Save the softmax result for backward
        ctx.save_for_backward(softmax)
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the softmax function.
        Arguments:
        - ctx: Context with saved tensors from forward pass.
        - grad_output: Gradient of the loss with respect to the output.
        Returns:
        - grad_input: Gradient of the loss with respect to the input.
        """
        # Retrieve saved softmax
        (softmax,) = ctx.saved_tensors
        
        # Compute gradient using Jacobian
        grad_input = softmax * (grad_output - (grad_output * softmax).sum(dim=-1, keepdim=True))
        return grad_input
