import numpy as np

from dataclasses import dataclass


@dataclass
class LogisticRegression:
    epochs: int
    learning_rate: float
    threshold: float
    logging: bool

    def sigmoid(self, predictions: np.ndarray) -> np.ndarray:
        """The numerically stable implementation of the Sigmoid activation function."""

        neg_mask = predictions < 0
        pos_mask = ~neg_mask

        zs = np.empty_like(predictions)
        zs[neg_mask] = np.exp(predictions[neg_mask])
        zs[pos_mask] = np.exp(-predictions[pos_mask])

        res = np.ones_like(predictions)
        res[neg_mask] = zs[neg_mask]

        return res / (1 + zs)

    def mean_log_loss(self, predictions: np.ndarray, labels: np.ndarray) -> np.float32:
        """Computes the mean Cross Entropy Loss (in binary classification, also called Log-loss)."""

        return -(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)).mean()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the Logistic Regression model."""

        num_samples, num_features = features.shape
        self.weights, self.bias = np.zeros(num_features), 0

        for epoch in range(self.epochs):
            prediction = self.sigmoid(features.dot(self.weights) + self.bias)
            difference = prediction - labels

            d_weights = features.T.dot(difference) / num_samples
            d_bias = difference.sum() / num_samples

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            if self.logging:
                print(f"Mean Log-loss [{epoch}]: {self.mean_log_loss(prediction, labels):.3f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""

        return np.where(self.sigmoid(features.dot(self.weights) + self.bias) < self.threshold, 0, 1)
