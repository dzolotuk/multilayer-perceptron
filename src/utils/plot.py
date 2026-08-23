import matplotlib.pyplot as plt


def plot_history(history):
	epochs = range(1, len(history["loss"]) + 1)

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)

	plt.plot(epochs, history["loss"], label="Train Loss")
	plt.plot(epochs, history["val_loss"], label="Validation Loss")

	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss vs Epochs")
	plt.legend()

	plt.subplot(1, 2, 2)

	plt.plot(epochs, history["acc"], label="Train Accuracy")
	plt.plot(epochs, history["val_acc"], label="Validation Accuracy")

	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Epochs")
	plt.legend()

	plt.tight_layout()
	plt.show()