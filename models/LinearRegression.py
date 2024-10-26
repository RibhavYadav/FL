from models.model_template import Model
import tensorflow as tf


class LinearRegression(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=1000, verbose=False):
        # Initialises Linear Regression model.
        super().__init__(weights, bias, learning_rate, iterations, verbose)
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else None
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else None
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose

    def predict(self, x):
        # Predicts output based on current weights and bias.
        return x @ self.weights + self.bias

    def compute_loss(
            self, y_true, y_pred):
        # Calculated the mean square error.
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        # Converts given data to tensorflow's tensor type.
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # Initialises weights and bias if not given during object creation.
        if self.weights is None:
            self.weights = tf.Variable(tf.random.normal([x_train.shape[1], 1]), dtype=tf.float32)
        if self.bias is None:
            self.bias = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

        # Gradient descent for optimisation of weights.
        for iteration in range(self.iterations + 1):

            # Tensorflow's class to calculate gradient descent.
            with tf.GradientTape() as gradient_descent:
                y_pred = self.predict(x_train)
                loss = self.loss_function(y_train, y_pred)

            # Calculates partial derivatives using tensorflow's automatic differentiation.
            d_weights, d_bias = gradient_descent.gradient(loss, [self.weights, self.bias])

            # Applies the gradient calculated on the weights and bias.
            self.weights.assign_sub(self.lr * d_weights)
            self.bias.assign_sub(self.lr * d_bias)

            # Prints loss, weights and bias.
            if self.verbose and iteration % (self.iterations // 10) == 0:
                print(
                    f"Epoch: {iteration}\nLoss: {loss.numpy().flatten()}\nWeights: {self.weights.numpy().flatten()}\nBias: {self.bias.numpy().flatten()}\n")

        # Prints the final weights and bias
        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}\nFinal Bias: {self.bias.numpy().flatten()} ")
