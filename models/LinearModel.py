import tensorflow as tf

from models.model_template import Model


class LinearRegression(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=200, verbose=False):
        # Initialize the parent class with all shared parameters
        super().__init__(weights, bias, learning_rate, iterations, verbose)

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

        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}, Final Bias: {self.bias.numpy()}")


class OptimisedSGD(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=200, verbose=False):
        # Initialize the base Model class
        super().__init__(weights, bias, learning_rate, iterations, verbose)

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


class UnoptimisedSGD(Model):
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=200, verbose=False):
        # Initialize the base Model class
        super().__init__(weights, bias, learning_rate, iterations, verbose)

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
