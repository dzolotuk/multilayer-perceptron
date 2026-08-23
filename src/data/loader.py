import pandas as pd
import numpy as np

def load_dataset(path, with_labels=False):
	df = pd.read_csv(path, header=None)

	if with_labels:
		y_raw = df.iloc[:, 1].values
		y = np.array([1 if v == "M" else 0 for v in y_raw])
		y = np.eye(2)[y]

		X = df.iloc[:, 2:].to_numpy(dtype=np.float64)
		return X, y

	X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
	return X