import numpy as np

def binary_crossentropy(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	return -np.mean(
		y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
	)

def binary_crossentropy_derivative(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	return (y_pred - y_true) / (y_pred * (1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_derivative(y_true, y_pred):
	return (y_pred - y_true)