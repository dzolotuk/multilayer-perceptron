import numpy as np

class DenseLayer:
	def __init__(self, input_size, output_size, activation, activation_derivative):
		self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
		self.bias = np.zeros((1, output_size))

		self.activation = activation
		self.activation_derivative = activation_derivative

	def forward(self, x):
		self.input = x
		self.z = np.dot(x, self.weights) + self.bias
		self.output = self.activation(self.z)
		return self.output

	def backward(self, grad_output, learning_rate):
		grad_activation = grad_output * self.activation_derivative(self.z)

		grad_weights = np.dot(self.input.T, grad_activation)
		grad_bias = np.sum(grad_activation, axis=0, keepdims=True)
		grad_input = np.dot(grad_activation, self.weights.T)

		self.weights -= learning_rate * grad_weights
		self.bias -= learning_rate * grad_bias

		return grad_input