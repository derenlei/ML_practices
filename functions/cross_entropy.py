import torch

class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        """
        Initializes the Cross-Entropy Loss class.
        
        Args:
            reduction (str): Specifies the reduction type - 'mean', 'sum', or 'none'.
        """
        self.reduction = reduction
        self.probs = None  # To store softmax probabilities for backward pass
    
    def forward(self, logits, targets):
        """
        Forward pass: Computes the cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs (logits) of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size,).
        
        Returns:
            torch.Tensor: Computed loss (scalar or per-sample depending on reduction).
        """
        # Apply softmax to logits to compute probabilities
        exp_logits = torch.exp(logits)
        self.probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
        # Gather the probabilities corresponding to the true classes
        true_class_probs = self.probs[torch.arange(targets.shape[0]), targets]
        
        # Compute negative log likelihood
        log_probs = -torch.log(true_class_probs)
        
        # Apply reduction
        if self.reduction == 'mean':
            return log_probs.mean()
        elif self.reduction == 'sum':
            return log_probs.sum()
        elif self.reduction == 'none':
            return log_probs
        else:
            raise ValueError("Invalid reduction type. Choose from 'mean', 'sum', or 'none'.")
    
    def backward(self, logits, targets):
        """
        Backward pass: Computes the gradient of the loss w.r.t. logits.
        
        Args:
            logits (torch.Tensor): Raw model outputs (logits) of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size,).
        
        Returns:
            torch.Tensor: Gradients w.r.t. logits, shape (batch_size, num_classes).
        """
        # Initialize gradients as probabilities
        grad = self.probs.clone()
        
        # Subtract 1 from the probabilities of the true class
        grad[torch.arange(targets.shape[0]), targets] -= 1
        
        # Divide by batch size for mean reduction
        if self.reduction == 'mean':
            grad /= targets.shape[0]
        
        return grad
