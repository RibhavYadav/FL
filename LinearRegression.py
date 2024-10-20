import numpy as np
import tensorflow as tf


class LinearRegression:
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=1000, verbose=False):
        self.weights = weights
        self.bias = bias
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose

    def predict(self, x):
        return x @ self.weights + self.bias

    def loss_function(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        if self.weights is None:
            self.weights = np.random.randn(x_train.shape[1])
        if self.bias is None:
            self.bias = np.random.randn(1)

        for iteration in range(self.iterations):
            y_pred = self.predict(x_train.transpose)
            loss = self.loss_function(y_train, y_pred)

            tape = tf.GradientTape()
            d_weights, d_bias = tape.gradient(loss, [self.weights, self.bias])

            self.weights -= self.lr * d_weights
            self.bias -= self.lr * d_bias

            if self.verbose and iteration % 100 == 0:
                print(f"Epoch: {iteration}\nLoss: {loss}\nWeights: {self.weights}\nBias: {self.bias} ")
