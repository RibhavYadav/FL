from client_side import Client
from typing import Type, List
from models.model_template import Model
import tensorflow as tf


class Server:
    def __init__(self, model: Type[Model], x, y, weights=None, bias=None, clients: List[Type[Client]] = None):
        # Initialize model, weights, and bias for the server
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else tf.Variable(tf.zeros((x.shape[1], 1), dtype=tf.float32))
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else tf.Variable(tf.zeros((1,), dtype=tf.float32))
        self.model = model(weights=self.weights, bias=self.bias)
        self.x, self.y = x, y
        self.connected_clients = clients

    def update_clients(self):
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def aggregate(self):
        # Accumulators for weights and bias
        total_w = tf.Variable(tf.zeros_like(self.weights), dtype=tf.float32)
        total_b = tf.Variable(tf.zeros_like(self.bias), dtype=tf.float32)

        # Collect weights and biases from clients
        for client in self.connected_clients:
            client.train_local(self.x, self.y)
            w, b = client.send_local()
            total_w.assign_add(w)
            total_b.assign_add(b)

        # Average the weights and bias
        num_clients = len(self.connected_clients)
        if num_clients > 0:
            self.weights.assign(total_w / num_clients)
            self.bias.assign(total_b / num_clients)

        # Update clients with the new weights and bias
        self.update_clients()
