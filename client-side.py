from models import model_template

"""
'Client' side implementation of the devices connected in Federated Learning
Client class with functions for it's working:
    - receive_update: receives the parameters from the global model along with other parameters
        - weights: array like
        - batch_size: minimum number of rows in a batch
    - train_local: initialise model and train it for a fixed batch size
        - initialise_model: start a local Linear Regression model
        - split_data: split the data into the given number of batch size
        - train_model: train the model on the local data for 1 batch
    - send_local: sends the updated model parameters back to the global model
"""


class Client:
    def __init__(self, learning_model: model_template, weights=None, bias=None, batch_size=64):
        self.model = learning_model
        self.weights = weights
        self.bias = bias
        self.batch_size = batch_size
        self.batches_done = 0

    def receive_update(self, weights, bias):
        self.weights, self.bias = weights, bias

    def train_local(self, x_train, y_train):
        split = self.batches_done * self.batch_size
        x_batch = x_train[split:split + self.batch_size]
        y_batch = y_train[split:split + self.batch_size]

        model = self.model(self.weights, self.bias)
        model.fit(x_batch, y_batch)

        self.batches_done += 1

    def send_local(self):
        return self.weights, self.bias
