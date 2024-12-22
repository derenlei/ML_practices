import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = scores - np.max(scores)
	o = scores - np.log(np.sum(np.exp(scores), axis=-1))
	return o

assert log_softmax([1, 2, 3]) == [-2.4076, -1.4076, -0.4076]
