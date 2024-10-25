import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and scale the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define a simple linear model (y = XW + b)
class LinearModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([X_train.shape[1], 1]))
        self.b = tf.Variable(tf.random.normal([1]))

    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b

model = LinearModel()

# Mean squared error (MSE) loss function
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Use SGD as the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Train the model
def train_step(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = compute_loss(y, predictions)
    gradients = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    return loss

# Training loop
epochs = 100
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_tf[start:end]
        y_batch = y_train_tf[start:end]
        
        batch_loss = train_step(model, X_batch, y_batch)
        epoch_loss += batch_loss

    print(f'Epoch {epoch+1}, Loss: {epoch_loss.numpy() / n_batches}')

# Evaluate the model
predictions = model(X_test_tf)
test_loss = compute_loss(y_test_tf, predictions)
print(f'Test Loss: {test_loss.numpy()}')
