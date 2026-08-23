import numpy as np

def encode_labels(y):
	return np.array([1 if v == 'M' else 0 for v in y]).reshape(-1, 1)

def normalize(X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	return (X - mean) / (std + 1e-8)