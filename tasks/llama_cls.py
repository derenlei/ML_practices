
Here's how you can implement LlamaForSequenceClassification from scratch using PyTorch. We'll leverage the transformers library to load the pre-trained LLaMA model's base architecture and add a classification head for sequence classification tasks.

Implementation: LlamaForSequenceClassification
python
Copy code
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig
import torch.nn as nn

class CrossEntropyLossFromScratch(nn.Module):
    def __init__(self):
        super(CrossEntropyLossFromScratch, self).__init__()

    def forward(self, logits, targets):
        """
        Computes the cross-entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size,).

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        # Step 1: Apply softmax to convert logits to probabilities
        probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
        
        # Step 2: Pick the probabilities of the correct classes
        # Gather correct class probabilities using advanced indexing
        correct_class_probs = probs[torch.arange(len(targets)), targets]
        
        # Step 3: Compute the negative log likelihood
        log_probs = -torch.log(correct_class_probs)
        
        # Step 4: Return the mean loss
        loss = torch.mean(log_probs)

        # Save variables for backward
        self.saved_probs = probs
        self.saved_targets = targets

        return loss

    def backward(self, logits):
        """
        Computes the gradient of the loss with respect to the logits.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Gradient of the loss with respect to logits.
        """
        # Get saved probabilities and targets
        probs = self.saved_probs
        targets = self.saved_targets  

        one_hot_target = torch.zeros_like(probs)
        one_hot_target[torch.arange(len(targets)), targets] = 1

        grad = (probs - one_hot_targets) / len(targets)  # Average gradient over the batch
        
        return grad

# Example Usage
if __name__ == "__main__":
    # Simulated logits and targets
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
    targets = torch.tensor([0, 1])  # True class indices

    # Custom loss function
    criterion = CrossEntropyLossFromScratch()
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item()}")

    # Backpropagation
    loss.backward()
    print(f"Gradients: {logits.grad}")


class LlamaForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes LLaMA model for sequence classification.

        Args:
            model_name (str): Pretrained model name or path (e.g., "meta-llama/Llama-2-7b").
            num_labels (int): Number of output labels for classification.
        """

        # Load the base LLaMA model
        self.config = LlamaConfig.from_pretrained(model_name)
        self.num_labels = num_labels
        self.llama = LlamaModel.from_pretrained(model_name)

        self.cls = nn.Linear(self.condfig.hidden_size, num_labels)
        self.loss_fn  = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels = None):
        """
        Forward pass for sequence classification.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length).
            labels (torch.Tensor, optional): Ground truth labels for supervised training.

        Returns:
            dict: A dictionary containing:
                - logits (torch.Tensor): Classification logits.
                - loss (torch.Tensor, optional): Loss value if labels are provided.
        """
        o = self.llama(input_ids, attention_mask)
        # logits = self.cls(0[:,-1,:]) # if left padding

        seq_len = attention_mask.sum(dim=1) - 1
        polled_output = o.hidden_states[torch.arange(input_ids.shape[0]), seq_len]
        logits = self.cls(polled_output)
        
        loss = self.loss_fn(logits, labels)
        
        output = {"logits": logits， "loss": loss}
        return output
