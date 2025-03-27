import torch

class HardMargin_SVM:
    def __init__(self, lr, epochs):
        """
        Initialize the HardMargin_SVM class.

        Parameters:
        - lr: Learning rate for gradient descent.
        - epochs: Number of epochs for training.
        """
        self.lr = lr
        self.epochs = epochs

    # Compute forward pass (linear decision boundary)
    def forward(self, x):
        """
        Compute the linear output (decision boundary).

        Parameters:
        - x: Input features (N, d).

        Returns:
        - Linear output (N,).
        """
        return x @ self.w + self.b

    def train(self, x, labels):
        """
        Train the SVM model using gradient descent.

        Parameters:
        - x: Input features (N, d).
        - labels: True labels (N,).
        """
        # Initialize weights and bias
        self.w = torch.zeros(x.shape[1], 1, requires_grad=False)
        self.b = torch.zeros(1, requires_grad=False)

        for epoch in range(self.epochs):
            # Compute outputs
            outputs = self.forward(x).squeeze()

            # Compute gradient of the hard-margin constraint
            # Hard Margin requires all points to satisfy y_i(w^T x_i + b) >= 1
            margins = labels * outputs
            mask = margins < 1  # Identify margin-violating samples

            # Compute gradients for margin-violating samples only
            grad_w = torch.mean(-labels[mask][:, None] * x[mask], dim=0)
            grad_b = torch.mean(-labels[mask]) if mask.any() else 0

            # Add regularization term (minimizing ||w||^2)
            grad_w += self.w

            # Update weights and bias
            self.w -= self.lr * grad_w.unsqueeze(1)
            self.b -= self.lr * grad_b

            # Compute loss (Optional: purely for monitoring purposes)
            loss = 0.5 * torch.sum(self.w ** 2)
            if mask.any():
                loss += torch.mean(1 - margins[mask])

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def predict(self, x):
        """
        Predict class labels for the input data.

        Parameters:
        - x: Input features (N, d).

        Returns:
        - Predicted labels (N,).
        """
        return torch.sign(self.forward(x))
