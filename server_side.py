from client_side import Client
from typing import Type, List
from models.model_template import Model
import tensorflow as tf


class Server:
    def __init__(self, model: Type[Model], x, y, weights=None, bias=None, clients: int = 3):
        # Initialize model, weights, and bias for the server
        if weights is not None:
            self.weights = tf.Variable(weights, dtype=tf.float32)
        else:
            tf.Variable(tf.zeros((x.shape[1], 1), dtype=tf.float32))
        if bias is not None:
            self.bias = tf.Variable(bias, dtype=tf.float32)
        else:
            tf.Variable(tf.zeros((1,), dtype=tf.float32))
        self.model = model(weights=self.weights, bias=self.bias)
        self.x, self.y = x, y
        self.connected_clients = [Client(model, self.weights, self.bias) for _ in range(clients)]

    def update_clients(self):
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def get_loss(self, y_true, x_test):
        return self.model.compute_loss(y_true, self.model.predict(x_test))

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
        clients = len(self.connected_clients)
        self.weights.assign(total_w / clients)
        self.bias.assign(total_b / clients)
        self.model.weights.assign(self.weights)
        self.model.bias.assign(self.bias)

        # Update clients with the new weights and bias
        self.update_clients()
