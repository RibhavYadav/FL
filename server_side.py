from client_side import Client
from typing import Type, List
from models.model_template import Model
import tensorflow as tf


class Server:
    def __init__(self, model: Type[Model], x, y, weights=None, bias=None, clients: List[Client] = None,
                 final_weights=False):
        # Initialize model, weights, and bias for the server
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else None
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else None
        self.model = model(weights=self.weights, bias=self.bias, final_weights=final_weights)
        self.x, self.y = x, y
        self.connected_clients = clients

    def update_clients(self):
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def aggregate(self):
        # Accumulators for weights and bias
        total_w = tf.zeros_like(self.weights)
        total_b = tf.zeros_like(self.bias)

        # Collect weights and biases from clients
        for client in self.connected_clients:
            client.train_local(self.x, self.y)
            w, b = client.send_local()
            total_w += tf.convert_to_tensor(w)
            total_b += tf.convert_to_tensor(b)

        # Average the weights and bias
        self.weights.assign(total_w / len(self.connected_clients))
        self.bias.assign(total_b / len(self.connected_clients))

        # Update clients with the new weights and bias
        self.update_clients()
