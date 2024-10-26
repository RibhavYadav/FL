import tensorflow as tf
from models.model_template import Model


class OptimisedSGD(Model):
    def __init__(self, learning_rate=0.001, iterations=1000, batch_size=32, verbose=False):
        super().__init__(learning_rate=learning_rate, iterations=iterations, verbose=verbose)
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def predict(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x_train, y_train):
        # Converts given data to TensorFlow tensors.
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # Initialize weights and bias based on the shape of the input features
        features = x_train.shape[1]
        self.weights = tf.Variable(tf.random.normal(shape=[features, 1]), dtype=tf.float32)
        self.bias = tf.Variable(tf.random.normal(shape=[1]), dtype=tf.float32)

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        # Create a TensorFlow dataset for batching
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(x_train)).batch(self.batch_size)

        # Training loop
        for epoch in range(self.iterations):
            epoch_loss = 0
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.predict(x_batch)
                    loss = self.compute_loss(y_batch, predictions)

                # Compute gradients
                gradients = tape.gradient(loss, [self.weights, self.bias])

                # Gradient clipping to prevent exploding gradients
                gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

                # Apply gradients
                optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))

                epoch_loss += loss.numpy()

            # Verbose output
            if self.verbose and (epoch % (self.iterations // 10) == 0 or epoch == self.iterations - 1):
                print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataset):.4f}')

        # Prints the final weights and bias.
        if self.verbose:
            print(f"Final Weights: {self.weights.numpy().flatten()}\nFinal Bias: {self.bias.numpy().flatten()} ")
