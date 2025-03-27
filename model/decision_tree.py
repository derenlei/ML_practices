import numpy as np
import torch

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # x (sample_num, features_num), y (label,)
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return torch.tensor([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or self._is_pure(y):
            return Node(value=self._calculate_leaf_value(y))

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return Node(value=self._calculate_leaf_value(y))

        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_split['feature'],
            threshold=best_split['threshold'],
            left=left_tree,
            right=right_tree
        )

    def _find_best_split(self, X, y):
        best_split = None
        best_impurity = float('inf')

        for feature in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                impurity = self._calculate_impurity(y[left_mask], y[right_mask])
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'impurity': impurity
                    }

        return best_split

    def _calculate_impurity(self, left, right):
        total_samples = len(left) + len(right)
        left_weight = len(left) / total_samples
        right_weight = len(right) / total_samples

        def gini(y):
            proportions = torch.bincount(y.long()) / len(y)
            return 1.0 - torch.sum(proportions ** 2)

        return left_weight * gini(left) + right_weight * gini(right)

    def _is_pure(self, y):
        return len(torch.unique(y)) == 1

    def _calculate_leaf_value(self, y):
        return torch.mode(y).values.item() if len(y) > 0 else 0

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Example Usage
if __name__ == "__main__":
    # Dataset (XOR problem for classification)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Train Decision Tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    # Predict
    predictions = tree.predict(X)
    print("Predictions:", predictions)
