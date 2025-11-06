import numpy as np
class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]  # bias term
        return self.activation_fn(z)
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                update = self.learning_rate * (target - output)
                self.weights[1:] += update * xi
                self.weights[0] += update  # bias update
# Example usage for AND logic gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND gate output
slp = SingleLayerPerceptron(input_size=2)
slp.train(X, y)
# Test predictions
for sample in X:
    print(f"Input: {sample}, Predicted: {slp.predict(sample)}")
