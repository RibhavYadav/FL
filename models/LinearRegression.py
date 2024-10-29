from models.model_template import Model
import tensorflow as tf


class LinearRegression(Model):
    def __init__(self, **kwargs):
        # Initialize the parent class with all shared parameters
        super().__init__(**kwargs)

    def predict(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        for epoch in range(self.iterations + 1):
            with tf.GradientTape() as tape:
                y_pred = self.predict(x_train)
                loss = self.compute_loss(y_train, y_pred)

            # Compute and apply gradients
            d_weights, d_bias = tape.gradient(loss, [self.weights, self.bias])
            self.weights.assign_sub(self.lr * d_weights)
            self.bias.assign_sub(self.lr * d_bias)

            # Verbose output for tracking progress
            if self.verbose and epoch % (self.iterations // 10) == 0:
                print(
                    f"Epoch: {epoch}, Loss: {loss.numpy().flatten()}, Weights: {self.weights.numpy().flatten()}, Bias: {self.bias.numpy()}")

        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}, Final Bias: {self.bias.numpy()}")
