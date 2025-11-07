#Implement Multi-Layer Perceptron (MLP) using Backpropagation Algorithm. 
import numpy as np
# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        # Initialize weights
        self.w_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))  
        # Biases
        self.b_hidden = np.zeros((1, hidden_size))
        self.b_output = np.zeros((1, output_size))
    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            # Forward pass
            hidden_input = np.dot(X, self.w_input_hidden) + self.b_hidden
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.w_hidden_output) + self.b_output
            final_output = sigmoid(final_input)
            # Backward pass
            error = y - final_output
            d_output = error * sigmoid_derivative(final_output)
            error_hidden = d_output.dot(self.w_hidden_output.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_output)
            # Update weights and biases
            self.w_hidden_output += hidden_output.T.dot(d_output) * self.learning_rate
            self.b_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.w_input_hidden += X.T.dot(d_hidden) * self.learning_rate
            self.b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
    def predict(self, X):
        hidden = sigmoid(np.dot(X, self.w_input_hidden) + self.b_hidden)
        output = sigmoid(np.dot(hidden, self.w_hidden_output) + self.b_output)
        return np.round(output)
# Example usage for XOR problem
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # XOR output
mlp = MultiLayerPerceptron(input_size=2, hidden_size=2, output_size=1)
mlp.train(X, y, epochs=10000)
# Test predictions and display actual vs predicted
print("Actual vs Predicted:")
for i in range(len(X)):
    sample = X[i].reshape(1, -1)  # Ensure it's 2D
    actual = y[i]
    predicted = mlp.predict(sample)
    print(f"Input: {sample.flatten()}, Actual: {actual[0]}, Predicted: {int(predicted[0][0])}")

