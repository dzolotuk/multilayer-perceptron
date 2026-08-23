from src.core.loss import categorical_crossentropy, categorical_crossentropy_derivative
from src.utils.metrics import precision, recall, f1_score

def accuracy(y_true, y_pred):
	y_pred = (y_pred > 0.5).astype(int)
	return (y_pred == y_true).mean()


def train_step(model, X, y, lr):
	# forward
	y_pred = model.forward(X)

	# loss
	loss = categorical_crossentropy(y, y_pred)

	# backward
	grad = categorical_crossentropy_derivative(y, y_pred)
	model.backward(grad, lr)

	acc = accuracy(y, y_pred)

	return loss, acc


def validate_step(model, X, y):
	y_pred = model.forward(X)

	loss = categorical_crossentropy(y, y_pred)
	acc = accuracy(y, y_pred)

	return loss, acc


def train(model, X_train, y_train, X_val, y_val, epochs, lr, patience=5, use_metrics=False):
	history = {
		"loss": [],
		"val_loss": [],
		"acc": [],
		"val_acc": []
	}

	if use_metrics:
		history.update({
			"precision": [],
			"val_precision": [],
			"recall": [],
			"val_recall": [],
			"f1": [],
			"val_f1": []
		})

	best_val_loss = float("inf")
	counter = 0

	for epoch in range(epochs):
		train_loss, train_acc = train_step(model, X_train, y_train, lr)
		val_loss, val_acc = validate_step(model, X_val, y_val)

		history["loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["acc"].append(train_acc)
		history["val_acc"].append(val_acc)

		if use_metrics:
			y_train_pred = model.predict(X_train)
			y_val_pred = model.predict(X_val)

			p_train = precision(y_train, y_train_pred)
			r_train = recall(y_train, y_train_pred)
			f1_train = f1_score(y_train, y_train_pred)

			p_val = precision(y_val, y_val_pred)
			r_val = recall(y_val, y_val_pred)
			f1_val = f1_score(y_val, y_val_pred)

			history["precision"].append(p_train)
			history["recall"].append(r_train)
			history["f1"].append(f1_train)

			history["val_precision"].append(p_val)
			history["val_recall"].append(r_val)
			history["val_f1"].append(f1_val)

		if use_metrics:
			print(
				f"epoch {epoch:03d} - "
				f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
				f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
				f"precision: {p_train:.4f} - val_precision: {p_val:.4f} - "
				f"recall: {r_train:.4f} - val_recall: {r_val:.4f} - "
				f"f1: {f1_train:.4f} - val_f1: {f1_val:.4f}"
			)
		else:
			print(
				f"epoch {epoch:03d} - "
				f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
				f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
			)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			counter = 0
		else: 
			counter += 1
		
		if counter >= patience:
			print(f"Early stopping at epoch {epoch}")
			break

	return history