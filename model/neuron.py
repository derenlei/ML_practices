import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
	w = initial_weights
	b = initial_bias
	n = len(labels)
	mse_values = []
	for e in range(epochs):
		preds = sigmoid(features @ w.T + b)
		mse = np.sum((labels-preds)**2) / n
		mse_values.append(round(mse, 4))
		dy =  -2/n * (labels - preds)
		dw = features.T @ (dy * preds * (1-preds))
		db = np.sum(dy * preds * (1-preds))
		w -= learning_rate * dw
		b -= learning_rate * db

		w = np.round(w, decimals= 4)
		b = np.round(b, 4)

	return w.tolist(), b, mse_values
