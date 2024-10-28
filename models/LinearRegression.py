from models.model_template import Model
import tensorflow as tf


class LinearRegression(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=1000, verbose=False):
        super().__init__(weights, bias, learning_rate, iterations, verbose)
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else None
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else None
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose

    def predict(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        if self.weights is None:
            self.weights = tf.Variable(tf.random.normal([x_train.shape[1], 1], stddev=0.01), dtype=tf.float32)
        if self.bias is None:
            self.bias = tf.Variable(tf.zeros([1]), dtype=tf.float32)

        for epoch in range(self.iterations + 1):
            with tf.GradientTape() as tape:
                y_pred = self.predict(x_train)
                loss = self.compute_loss(y_train, y_pred)

            d_weights, d_bias = tape.gradient(loss, [self.weights, self.bias])
            self.weights.assign_sub(self.lr * d_weights)
            self.bias.assign_sub(self.lr * d_bias)

            if self.verbose and epoch % (self.iterations // 10) == 0:
                print(
                    f"Epoch: {epoch}, Loss: {loss.numpy().flatten()}, Weights: {self.weights.numpy().flatten()}, Bias: {self.bias.numpy()}")

        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}, Final Bias: {self.bias.numpy()}")
