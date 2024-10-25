from client_side import Client
from typing import Type, List
from models.model_template import Model


class Server:
    def __init__(self, model: Type[Model], x, y, weights=None, bias=None, clients: List[Type[Client]] = None):
        # Initialize model, weights, and bias for the server
        self.weights = weights
        self.bias = bias
        self.model = model(weights=self.weights, bias=self.bias)
        self.x, self.y = x, y
        self.connected_clients = clients

    def update_clients(self):
        for client in self.connected_clients:
            client.receive_update(self.weights, self.bias)

    def aggregate(self):
        new_w, new_b = self.weights, self.bias
        for client in self.connected_clients:
            client.train_local(self.x, self.y)
            w, b = client.send_local()
            new_w += w
            new_b += b
        self.weights = new_w / len(self.connected_clients)
        self.bias = new_b / len(self.connected_clients)
        self.update_clients()
