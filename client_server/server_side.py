from client_server.client_side import Client
from typing import Type, List
from models.model_template import Model
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor


class Server:
    def __init__(self, model: Type[Model], x, y, weights=None, bias=None, clients: int = 3, batch_size: int = 64):
        # Initialize model, weights, and bias for the server
        if weights is not None:
            self.weights = tf.Variable(weights, dtype=tf.float32)
        else:
            tf.Variable(tf.zeros((x.shape[1], 1), dtype=tf.float32))
        if bias is not None:
            self.bias = tf.Variable(bias, dtype=tf.float32)
        else:
            tf.Variable(tf.zeros((1,), dtype=tf.float32))

        # tf.data.Dataset and iterator for batch processing
        self.dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        self.batch_iterator = iter(self.dataset)

        # Initialize the model and clients
        self.model = model(weights=self.weights, bias=self.bias)
        self.connected_clients = [Client(model, self.weights, self.bias, batch_size=batch_size) for _ in range(clients)]
        self.batch_size = batch_size
        self.batches_done = 0

    def update_clients(self):
        # Update each client
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def get_loss(self, y_true, x_test):
        # Calculate loss for the server model
        return self.model.compute_loss(y_true, self.model.predict(x_test))

    def aggregate(self):
        # Get the next batch of data
        x_batch, y_batch = next(self.batch_iterator)

        # Aggregators for weights and biases from clients
        total_w = tf.Variable(tf.zeros_like(self.weights), dtype=tf.float32)
        total_b = tf.Variable(tf.zeros_like(self.bias), dtype=tf.float32)

        # Parallelize client training
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(client.train_local, x_batch, y_batch)
                for client in self.connected_clients
            ]
            for future in futures:
                future.result()

        for client in self.connected_clients:
            w, b = client.send_local()
            total_w.assign_add(w)
            total_b.assign_add(b)

        num_clients = len(self.connected_clients)
        self.weights.assign(total_w / num_clients)
        self.bias.assign(total_b / num_clients)
        self.model.weights.assign(self.weights)
        self.model.bias.assign(self.bias)

        # Update clients with new aggregated weights
        self.update_clients()
