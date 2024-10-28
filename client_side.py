from models.model_template import Model
from typing import Type
import tensorflow as tf


class Client:
    def __init__(self, model: Type[Model], weights=None, bias=None, batch_size=64):
        # Initialize the client with the model, weights, and bias from the global model.
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else None
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else None
        self.model = model(weights=self.weights, bias=self.bias)
        self.batch_size = batch_size
        self.batches_done = 0

    def receive_update(self, weights, bias):
        # Update the weights and bias of the local model
        self.weights.assign(weights)  # Correctly assign the new weights
        self.bias.assign(bias)        # Correctly assign the new bias
        self.model.weights = self.weights  # Update the model's weights
        self.model.bias = self.bias        # Update the model's bias

    def train_local(self, x_train, y_train):
        # Train the local model on a batch of data
        split = self.batches_done * self.batch_size
        x_batch = x_train[split:split + self.batch_size]
        y_batch = y_train[split:split + self.batch_size]

        # Fit the model to the batch
        self.model.fit(x_batch, y_batch)

        # Update the local weights and bias after training
        self.weights.assign(self.model.weights)
        self.bias.assign(self.model.bias)

        self.batches_done += 1

    def send_local(self):
        # Return the current weights and bias
        return self.weights, self.bias
