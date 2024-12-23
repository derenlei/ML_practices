import numpy as np

from dataclasses import dataclass


@dataclass
class KNN:
    features: np.ndarray
    labels: np.ndarray
    k: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""

        num_samples, _ = features.shape

        predictions = np.empty(num_samples)
        for idx, feature in enumerate(features):
            distances = [np.linalg.norm(feature - train_feature) for train_feature in self.features]
            k_sorted_idxs = np.argsort(distances)[: self.k]
            most_common = np.bincount([self.labels[idx] for idx in k_sorted_idxs]).argmax()
            predictions[idx] = most_common

        return predictions
