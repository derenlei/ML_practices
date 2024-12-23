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
