import torch

class SoftMargin_SVM:
    def __init__(self, lr, epochs, C):
        """
        Initialize the SoftMargin_SVM class.

        Parameters:
        - lr: Learning rate for gradient descent.
        - epochs: Number of epochs for training.
        - C: Regularization parameter for the soft margin.
        """
        self.lr = lr
        self.epochs = epochs
        self.C = C  # Regularization parameter

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

    # Compute hinge loss
    def hinge_loss(self, outputs, labels):
        """
        Compute the hinge loss.

        Parameters:
        - outputs: Model outputs (N,).
        - labels: True labels (N,).

        Returns:
        - Hinge loss value.
        """
        # max(0, 1 - y (w^T x + b))
        margin_violations = torch.clamp(1 - labels * outputs, min=0)
        return torch.mean(margin_violations)

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

            # Compute hinge loss
            hinge_loss = self.hinge_loss(outputs, labels)

            # Compute total loss (hinge loss + regularization)
            loss_regularization = 0.5 * torch.sum(self.w ** 2)
            loss = hinge_loss + loss_regularization / self.C

            # Backpropagation (manual gradient computation)
            margins = 1 - labels * outputs
            mask = margins > 0  # Identify margin-violating samples

            # Compute gradients
            grad_w = torch.mean(-labels[:, None] * x * mask[:, None], dim=0) + self.w / self.C
            grad_b = torch.mean(-labels * mask)

            # Update weights and bias using gradient descent
            self.w -= self.lr * grad_w.unsqueeze(1)
            self.b -= self.lr * grad_b

            # Print loss every 10 epochs
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
