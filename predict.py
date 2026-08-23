import argparse
import numpy as np
import pandas as pd

from src.core.loss import categorical_crossentropy
from src.core.activations import sigmoid, sigmoid_derivative, softmax
from src.core.layer import DenseLayer
from src.core.network import NeuralNetwork


DROP_COLUMNS = ["id", "Unnamed: 32"]


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


def load_model(path):
	data = np.load(path, allow_pickle=True).item()

	mean = data["mean"]
	std = data["std"]
	layers_data = data["layers"]

	layers = []

	for i, layer_data in enumerate(layers_data):
		input_size = layer_data["weights"].shape[0]
		output_size = layer_data["weights"].shape[1]

		if i == len(layers_data) - 1:
			layer = DenseLayer(
				input_size,
				output_size,
				softmax,
				lambda x: np.ones_like(x)
			)
		else:
			layer = DenseLayer(
				input_size,
				output_size,
				sigmoid,
				sigmoid_derivative
			)

		layer.weights = layer_data["weights"]
		layer.bias = layer_data["bias"]

		layers.append(layer)

	model = NeuralNetwork(layers)

	return model, mean, std


def normalize(X, mean, std):
	return (X - mean) / std


def predict(model, X):
	probs = model.forward(X)
	preds = np.argmax(probs, axis=1)
	return preds, probs


def accuracy(y_true, y_pred):
	y_true_labels = np.argmax(y_true, axis=1)
	return (y_true_labels == y_pred).mean()


def main():
	parser = argparse.ArgumentParser(description="MLP prediction")

	parser.add_argument("--model", required=True, help="Path to saved model")
	parser.add_argument("--input", help="CSV file for prediction")
	parser.add_argument("--val", action="store_true", help="Use saved validation set")

	args = parser.parse_args()

	model, mean, std = load_model(args.model)

	X = None
	y = None

	if args.val:
		X = np.load("data/processed/X_val.npy")
		y = np.load("data/processed/y_val.npy")

	elif args.input:
		X, y = load_dataset(args.input, with_labels=True)

	else:
		raise ValueError("Provide --input or --val")

	X = normalize(X, mean, std)

	preds, probs = predict(model, X)

	print("Predictions:")
	for i in range(len(preds)):
		label = "M" if preds[i] == 1 else "B"
		confidence = probs[i][preds[i]]

		print(f"{i}: {label} (confidence: {confidence:.4f})")

	if y is not None:
		loss = categorical_crossentropy(y, probs)
		acc = accuracy(y, preds)

		print(f"\nLoss: {loss:.4f}")
		print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
	main()