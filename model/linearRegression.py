class LinearRegression:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def learn(self, dataset, labels):
        self.weights = torch.zeros(dataset.shape[-1], 1)
        self.bias = torch.tensor(0)
        
        for epoch in range(self.epochs):
            preds = self.predict(dataset)
            # loss = (labels - preds)**2.mean(dim=0)
            dw = -2/dataset.shape[0] * dataset.T @ (labels.unsqueeze(1) - preds)
            db = -2/dataset.shape[0] * (labels.unsqueeze(1) - press).sum(dim=0)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, dataset):
        output = dataset @ self.weights + self.bias
        return output


# -----------------numpy-----------------------
import numpy as np

from dataclasses import dataclass


@dataclass
class LinearRegression:
    epochs: int
    learning_rate: float
    logging: bool

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the Linear Regression model."""

        num_samples, num_features = features.shape
        self.weights, self.bias = np.zeros(num_features), 0

        for epoch in range(self.epochs):
            residuals = labels - self.predict(features)

            d_weights = -2 / num_samples * features.T.dot(residuals)
            d_bias = -2 / num_samples * residuals.sum()

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            if self.logging:
                print(f"MSE Loss [{epoch}]: {(residuals ** 2).mean():.3f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""
        return features @ self.weights + bias
