import tensorflow as tf
import pandas as pd
from server_side import Server
from models.LinearModel.LinearRegression import LinearRegression
from models.LinearModel.OptimisedSGD import OptimisedSGD
from models.LinearModel.UnoptimisedSGD import UnoptimisedSGD
import matplotlib.pyplot as plt

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

linear_server = Server(LinearRegression, x_train, y_train, weights=initial_weights, bias=initial_bias)
usgd_server = Server(UnoptimisedSGD, x_train, y_train, weights=initial_weights, bias=initial_bias)
osgd_server = Server(OptimisedSGD, x_train, y_train, weights=initial_weights, bias=initial_bias, clients=1)

linear_loss = [linear_server.get_loss(y_test, x_test).numpy()]
usgd_loss = [usgd_server.get_loss(y_test, x_test).numpy()]
osgd_loss = [osgd_server.get_loss(y_test, x_test).numpy()]

epochs = 2
for i in range(epochs):
    print(f"Epoch: {i + 1}")
    linear_server.aggregate(), usgd_server.aggregate(), osgd_server.aggregate()
    linear_loss.append(linear_server.get_loss(y_test, x_test).numpy())
    usgd_loss.append(usgd_server.get_loss(y_test, x_test).numpy())
    osgd_loss.append(osgd_server.get_loss(y_test, x_test).numpy())

print(linear_loss)
print(usgd_loss)
print(osgd_loss)

loss = [linear_loss, usgd_loss, osgd_loss]
titles = ["Linear Regression", "UnoptimisedSDG", "OptimisedSDG"]
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.plot(range(1, len(loss[i]) + 1), loss[i], marker="o")
    plt.title(titles[i])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
plt.legend()
plt.show()
