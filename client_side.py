from models import model_template


class Client:
    def __init__(self, learning_model: model_template, weights=None, bias=None, batch_size=64):
        # Initialises the client with the model, weights and bias from the global model.
        self.model = learning_model
        self.weights = weights
        self.bias = bias
        self.batch_size = batch_size
        self.batches_done = 0

    def receive_update(self, weights, bias):
        # Updates the weights and bias of the local model.
        self.weights, self.bias = weights, bias

    def train_local(self, x_train, y_train):
        # Trains the local model on a batch of data
        split = self.batches_done * self.batch_size
        x_batch = x_train[split:split + self.batch_size]
        y_batch = y_train[split:split + self.batch_size]

        model = self.model(self.weights, self.bias)
        model.fit(x_batch, y_batch)

        self.batches_done += 1

    def send_local(self):
        return self.weights, self.bias
