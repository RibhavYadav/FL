import tensorflow as tf
from models.model_template import Model


class UnoptimisedSGD(Model):
    def __init__(self, **kwargs):
        # Initialize the parent class with all shared parameters
        super().__init__(**kwargs)

    def predict(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        # Converts given data to TensorFlow tensors.
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        # Training loop
        for epoch in range(self.iterations + 1):
            with tf.GradientTape() as tape:
                y_pred = self.predict(x_train)
                loss = self.compute_loss(y_train, y_pred)

            # Compute and apply gradients
            gradients = tape.gradient(loss, [self.weights, self.bias])
            optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))

            # Verbose output for tracking progress
            if self.verbose and epoch % (self.iterations // 10) == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.numpy().flatten()}')

        # Final verbose output
        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}\nFinal Bias: {self.bias.numpy().flatten()} ")
