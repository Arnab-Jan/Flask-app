import numpy as np
# ... (all the Neural_Network class code) ...
class Neural_Network:
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lr = learning_rate
    self.weights1 = np.random.randn(self.input_size, self.hidden_size)
    self.weights2 = np.random.randn(self.hidden_size, self.output_size)
    self.bias1 = np.zeros((1, self.hidden_size))
    self.bias2 = np.zeros((1, self.output_size))
  def forward(self, X):
    self.z = np.dot(X, self.weights1) + self.bias1
    self.a = self.sigmoid(self.z)
    self.z2 = np.dot(self.a, self.weights2) + self.bias2
    self.a2 = self.sigmoid(self.z2)
    return self.a2
  def sigmoid(self, s):
    s = np.clip(s, -500, 500)
    return 1/(1 + np.exp(-s))
  def sigmoidPrime(self, s):
    return s * (1 - s)
  def backward(self, X, y, output):
    self.output_error = y - output
    self.output_delta = self.output_error * self.sigmoidPrime(output)
    self.z2_error = self.output_delta.dot(self.weights2.T)
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.a)
    self.weights1 += self.lr * X.T.dot(self.z2_delta)
    self.weights2 += self.lr * self.a.T.dot(self.output_delta)
    self.bias1 += self.lr * np.sum(self.z2_delta, axis=0, keepdims=True)
    self.bias2 += self.lr * np.sum(self.output_delta, axis=0, keepdims=True)
  def train(self, X, y, epochs):
    for i in range(epochs):
      output = self.forward(X)
      self.backward(X, y, output)
  def predict(self, X):
    output = self.forward(X)
    return output
  def loss(self, X, y):
    output = self.forward(X)
    mse = np.mean((y - output)**2)
    return mse
  def fit(self, X, y, epochs=1000):
    self.train(X, y, epochs)