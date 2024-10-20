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
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        if self.weights is None:
            self.weights = tf.Variable(tf.random.normal([x_train.shape[1], 1]), dtype=tf.float32)
        if self.bias is None:
            self.bias = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

        for iteration in range(self.iterations):
            gradient_descent = tf.GradientTape()

            y_pred = self.predict(x_train)
            loss = self.loss_function(y_train, y_pred)

            d_weights, d_bias = gradient_descent.gradient(loss, [self.weights, self.bias])
            self.weights.assign_sub(self.lr * d_weights)
            self.bias.assign_sub(self.lr * d_bias)

            if self.verbose and iteration % 100 == 0:
                print(f"Epoch: {iteration}\nLoss: {loss}\nWeights: {self.weights}\nBias: {self.bias} ")

        print(f"Weights: {self.weights}\nBias: {self.bias} ")


LR = LinearRegression(verbose=True)

np.random.seed(42)
X_train = np.random.rand(100, 1)
y_t = 3 * X_train + 2 + np.random.randn(100, 1) * 0.1

LR.fit(X_train, y_t)
