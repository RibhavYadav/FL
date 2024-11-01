import tensorflow as tf

"""Base model template"""


class Model:
    def __init__(self, weights=None, bias=None, learning_rate=0.01, iterations=200, verbose=False):
        """Initialises the model"""
        self.weights = tf.Variable(weights, dtype=tf.float32) if weights is not None else None
        self.bias = tf.Variable(bias, dtype=tf.float32) if bias is not None else None
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        pass

    def compute_loss(self, y_true, y_pred):
        """Computes mean squared error"""
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def predict(self, x):
        """Predicts output from current weights and bias"""
        return tf.matmul(x, self.weights) + self.bias

    def fit(self, x_train, y_train):
        """Fits the model to the training data"""
        pass
