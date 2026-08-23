import numpy as np

def train_test_split(X, y, test_size=0.2, seed=42):
	np.random.seed(seed)

	indices = np.arange(len(X))
	np.random.shuffle(indices)

	split = int(len(X) * (1 - test_size))

	train_idx = indices[:split]
	val_idx = indices[split:]

	return X[train_idx], X[val_idx], y[train_idx], y[val_idx]