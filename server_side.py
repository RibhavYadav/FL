from client_side import Client
from typing import Type, List
from models.model_template import Model


class Server:
    def __init__(self, model: Type[Model], weights=None, bias=None, clients: List[Type[Client()]] = None):
        # Initialize model, weights, and bias for the server
        self.weights = weights
        self.bias = bias
        self.model = model(weights=self.weights, bias=self.bias)
        self.connected_clients = clients

    def broadcast_parameters(self):
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def aggregate(self):
        new_w, new_b = self.weights, self.bias
        for client in self.connected_clients:
            w, b = client.send_local()
            new_w += w
            new_b += b
        self.weights = new_w / len(self.connected_clients)
        self.bias = new_b / len(self.connected_clients)
        self.broadcast_parameters()
