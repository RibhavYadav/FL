import tensorflow as tf
from models.model_template import Model


class OptimisedSGD(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=200, verbose=False):
        # Initialize the base Model class
        super().__init__(weights, bias, learning_rate, iterations, verbose)

    def predict(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_batch, y_batch):
        # Ensure that x_batch and y_batch are tensors
        x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        # Training step on the single batch
        for epoch in range(self.iterations + 1):
            with tf.GradientTape() as tape:
                predictions = self.predict(x_batch)
                loss = self.compute_loss(y_batch, predictions)

            # Compute gradients
            gradients = tape.gradient(loss, [self.weights, self.bias])
            # Create an optimizer and apply gradients
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
            optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))

        # Verbose output for batch loss
        if self.verbose:
            print(f'Batch Loss: {loss.numpy():.4f}')
