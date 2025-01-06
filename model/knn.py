import torch
import torch.nn.functional as F

class KNN:
    def __init__(self, features, labels, k):
        self.k = k
        self.features = features
        self.labels = labels

    def predict(self, queries):
        # queries (num_queries, n_features)
        # torch.cdist(query, data, p=2)
        # same as 
        # query.unsqueeze(1)  # Shape: (n_queries, 1, n_features)
        # data.unsqueeze(0)    # Shape: (1, n_data, n_features)
        # (n_queries, n_data)
        dist = torch.sqrt(((query.squeeze(1) - self.features(0))**2).sum(-1))
        topks = torch.argmin(dist, dim=-1)[:self.k]
        knn_labels = labels[topks] # (n_queries, labels)
        predicted_labels = torch.mode(knn_labels, dim=-1).values
        return predicted_labels

# -----------------------------numpy-version-----------------------
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
