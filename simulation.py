import tensorflow as tf
import pandas as pd
from client_side import Client
from server_side import Server
from models.LinearRegression import LinearRegression
from models.OptimisedSGD import OptimisedSGD
from models.UnoptimisedSGD import UnoptimisedSGD

data = pd.read_csv("wine+quality/winequality-red.csv", delimiter=';')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Manual feature scaling (standardization)
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)
x = (x - x_mean) / x_std

# Split data into training and testing sets
row, col = x.shape
split = int(0.75 * row)
x_train, x_test = tf.convert_to_tensor(x[:split], dtype=tf.float32), tf.convert_to_tensor(x[split:], dtype=tf.float32)
y_train, y_test = tf.convert_to_tensor(y[:split], dtype=tf.float32), tf.convert_to_tensor(y[split:], dtype=tf.float32)

initial_weights = tf.Variable(tf.random.normal((col, 1), stddev=0.01))
initial_bias = 0
print("Initial: ", initial_weights.numpy().flatten(), initial_bias)
server = Server(LinearRegression, x_train, y_train, weights=initial_weights, bias=initial_bias)
print(server.model.compute_loss(y_test, server.model.predict(x_test)))

for i in range(3):
    server.aggregate()
    print(server.model.compute_loss(y_test, server.model.predict(x_test)))
