import numpy as np


def euclidean_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Euclidean distance between two tensors."""
    # return np.linalg.norm(t1-t2)
    return np.sqrt(np.sum((t1 - t2) ** 2))


def manhattan_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Manhattan distance between two tensors."""
    
    return np.sum(np.abs(t1-t2))

# ----------------pair-wise-cosine-similarity-----------------------------------------------
# each value 
def cosine_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    if np.count_nonzero(t1) == 0 or np.count_nonzero(t2) == 0:
        raise ValueError("Cosine distance is undefined for zero-tensors")
    return (t1 @ t2.T) / (np.linalg.norm(t1) * np.linalg.norm(t2))


# if a [N,D], b [N,D], it will be N * N similarity scores
def cosine_similarity(a, b):
    norm_a = torch.sqrt(torch.sum(a ** 2, dim=-1))
    norm_b = torch.sqrt(torch.sum(b ** 2, dim=-1))
    return a @ b.T / (norm_a * norm_b)


#--------------------element-wise-cosine-similarity---------------------------------------
# if a [N,D], b [N,D], it will be (N,) similarity scores
def cosine_similarity(a, b):
    # If a and b are single 1-D vectors:
    norm_a = torch.sqrt((a ** 2).sum())
    norm_b = torch.sqrt((b ** 2).sum())
    return (a * b).sum(dim=1) / (norm_a * norm_b)

# or batched along dim=0:
def batched_cosine_similarity(a, b):
    # a, b shape: [N, D]
    norm_a = torch.sqrt((a ** 2).sum(dim=-1))        # shape: [N]
    norm_b = torch.sqrt((b ** 2).sum(dim=-1))        # shape: [N]
    dot_prod = (a * b).sum(dim=-1)                   # shape: [N]
    return dot_prod / (norm_a * norm_b)              # shape: [N]
