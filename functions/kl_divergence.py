import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl_div = term1 + term2 - 0.5
    return kl_div

#---------------------------------------------------
class KLDivergenceDiscrete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q):
        """
        Forward pass for KL divergence.
        Args:
            p (torch.Tensor): True distribution (batch_size, vocab_size).
            q (torch.Tensor): Predicted distribution (batch_size, vocab_size).
        Returns:
            torch.Tensor: KL divergence for each example in the batch.
        """
        # Ensure numerical stability by clamping probabilities
        epsilon = 1e-9
        p = torch.clamp(p, min=epsilon)
        q = torch.clamp(q, min=epsilon)
        
        # Save tensors for backward
        ctx.save_for_backward(p, q)

        # Compute KL divergence
        kl_div = torch.sum(p * torch.log(p / q), dim=1)
        return kl_div

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for KL divergence.
        Args:
            grad_output (torch.Tensor): Gradients from the next layer (batch_size,).
        Returns:
            Tuple: Gradients with respect to (p, q).
        """
        p, q = ctx.saved_tensors

        # Gradient of KL divergence w.r.t. p: log(p / q) + 1
        grad_p = grad_output.unsqueeze(1) * (torch.log(p / q) + 1)

        # Gradient of KL divergence w.r.t. q: -p / q
        grad_q = grad_output.unsqueeze(1) * (-p / q)

        return grad_p, grad_q


#---------------------------------------------------
import torch
import torch.nn.functional as F

class KLDivergenceWithSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        """
        Forward pass for KL Divergence with softmax.
        Args:
            logits (torch.Tensor): Logits (raw scores before softmax), shape (batch_size, num_classes).
            target (torch.Tensor): True distribution (labels), shape (batch_size, num_classes).
        Returns:
            torch.Tensor: KL divergence loss for the batch.
        """
        # Apply softmax to logits to get predicted distribution
        softmax_probs = F.softmax(logits, dim=-1)
        
        # Save for backward pass
        ctx.save_for_backward(softmax_probs, target)
        
        kl_div = torch.sum(target * torch.log(target / softmax_probs), dim=-1)
        return kl_div.mean()  # Return mean loss over batch

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for KL Divergence with softmax.
        Args:
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output.
        Returns:
            Tuple[torch.Tensor, None]: Gradients w.r.t. logits, and None for target (non-trainable).
        """
        softmax_probs, target = ctx.saved_tensors
        grad = target - softmax_probs

        softmax_probs, target = ctx.saved_tensors

        # Gradient of KL divergence loss w.r.t. logits
        grad_logits = (softmax_probs - target) / softmax_probs.size(0)  # Normalize by batch size
        return grad_output.unsqueeze(1) * grad_logits, None


# Alias for readability
kl_softmax_loss_fn = KLDivergenceWithSoftmax.apply

# Example Usage
if __name__ == "__main__":
    # Example logits (raw scores) and target (true distribution)
    logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]], requires_grad=True)
    target = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], requires_grad=False)

    # Forward pass
    loss = kl_softmax_loss_fn(logits, target)
    print("KL Divergence Loss:", loss.item())

    # Backward pass
    loss.backward()
    print("Gradient w.r.t. logits:", logits.grad)
