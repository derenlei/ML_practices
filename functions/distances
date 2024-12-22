import numpy as np


def euclidean_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Euclidean distance between two tensors."""
    # return np.linalg.norm(t1-t2)
    return np.sqrt(np.sum((t1 - t2) ** 2))


def manhattan_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Manhattan distance between two tensors."""
    
    return np.sum(np.abs(t1-t2))

def cosine_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    if np.count_nonzero(t1) == 0 or np.count_nonzero(t2) == 0:
        raise ValueError("Cosine distance is undefined for zero-tensors")
    return (t1 @ t2.T) / (np.linalg.norm(t1) * np.linalg.norm(t2))
