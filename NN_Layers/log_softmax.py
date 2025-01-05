import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = scores - np.max(scores)
	o = scores - np.log(np.sum(np.exp(scores), axis=-1))
	return o

assert log_softmax([1, 2, 3]) == [-2.4076, -1.4076, -0.4076]

class LogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        max_vals = x.max(dim=-1, keepdim=True).values
        shifted_x = x - max_vals
        softmax = torch.exp(shifted_x) / torch.exp(shifted_x).sum(dim=-1, keepdim=True)
        log_softmax = shifted_x - torch.log(softmax.sum(dim=-1, keepdim=True))
        
        # Save softmax for backward computation
        ctx.save_for_backward(softmax)
        return log_softmax

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (softmax,) = ctx.saved_tensors
        return grad_output - softmax * grad_output.sum(dim-1, keepdim=True)
	
