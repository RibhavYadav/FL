"""Base model template"""


class Model:
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=1000, verbose=False,
                 final_weights=False):
        """Initialises the model"""
        pass

    def predict(self, x):
        """Predicts output from current weights and bias"""
        pass

    def fit(self, x_train, y_train):
        """Fits the model to the training data"""
        pass
