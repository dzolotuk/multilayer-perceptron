class NeuralNetwork:
	def __init__(self, layers):
		self.layers = layers

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, loss_grad, learning_rate):
		for layer in reversed(self.layers):
			loss_grad = layer.backward(loss_grad, learning_rate)

	def predict(self, X):
		output = self.forward(X)
		return (output > 0.5).astype(int)