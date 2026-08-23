import argparse
import numpy as np

from src.core.activations import sigmoid, sigmoid_derivative, softmax
from src.core.layer import DenseLayer
from src.core.network import NeuralNetwork
from src.data.loader import load_dataset
from src.data.split import train_test_split
from src.training.trainer import train
from src.utils.plot import plot_history


def normalize_from_train(X_train, X_val):
	mean = X_train.mean(axis=0)
	std = X_train.std(axis=0)

	std[std == 0] = 1.0

	return (
		(X_train - mean) / std,
		(X_val - mean) / std,
		mean,
		std
	)


def build_model(input_size, hidden_layers):
	layers = []
	prev = input_size

	for size in hidden_layers:
		layers.append(DenseLayer(prev, size, sigmoid, sigmoid_derivative))
		prev = size

	layers.append(DenseLayer(prev, 2, softmax, lambda x: 1))

	return NeuralNetwork(layers)


def save_model(model, path, hidden_layers, mean, std):
	data = {
		"hidden_layers": np.array(hidden_layers),
		"mean": mean,
		"std": std,
		"layers": []
	}

	for layer in model.layers:
		data["layers"].append({
			"weights": layer.weights,
			"bias": layer.bias
		})

	np.save(path, data, allow_pickle=True)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
	parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
	parser.add_argument("--layers", nargs="+", type=int, default=[24, 24], help="Hidden layer sizes")
	parser.add_argument("--model_path", default="saved_model.npy", help="Path to save trained model")
	parser.add_argument("--plot", action="store_true", help="Show training curves")
	parser.add_argument("--metrics", action="store_true", help="Enable precision/recall/f1 metrics")

	args = parser.parse_args()

	np.random.seed(42)

	X, y = load_dataset(args.dataset, with_labels=True)

	X_train, X_val, y_train, y_val = train_test_split(
		X, y,
		test_size=0.2,
		seed=42
	)

	print(f"x_train shape : {X_train.shape}")
	print(f"x_val shape : {X_val.shape}")

	X_train, X_val, mean, std = normalize_from_train(X_train, X_val)

	model = build_model(X_train.shape[1], args.layers)

	history = train(
		model,
		X_train,
		y_train,
		X_val,
		y_val,
		epochs=args.epochs,
		lr=args.lr,
		use_metrics=args.metrics
	)

	save_model(model, args.model_path, args.layers, mean, std)
	print(f"> saving model '{args.model_path}' to disk...")

	# FINAL RESULT SUMMARY
	final_train_loss = history["loss"][-1]
	final_val_loss = history["val_loss"][-1]
	final_train_acc = history["acc"][-1]
	final_val_acc = history["val_acc"][-1]

	print("\n=== FINAL RESULT ===")
	print(f"train_loss : {final_train_loss:.4f}")
	print(f"val_loss   : {final_val_loss:.4f}")
	print(f"train_acc  : {final_train_acc:.4f}")
	print(f"val_acc    : {final_val_acc:.4f}")

	if args.plot:
		plot_history(history)


if __name__ == "__main__":
	main()