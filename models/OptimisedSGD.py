import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and scale the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Scale the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale the target variable (reshape to 2D for StandardScaler)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Reshape to 2D, then flatten back

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

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

# Use SGD as the optimizer with a lower learning rate
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)  # Reduced learning rate

# Train the model
def train_step(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = compute_loss(y, predictions)
    gradients = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    return loss

# Training loop with more epochs
epochs = 300  # Increased number of epochs
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

    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch {epoch+1}, Loss: {epoch_loss.numpy() / n_batches}')

# Evaluate the model
predictions = model(X_test_tf)
test_loss = compute_loss(y_test_tf, predictions)
print(f'Test Loss: {test_loss.numpy()}')

# Optionally inverse transform the predictions and target back to original scale
predictions_original_scale = scaler_y.inverse_transform(predictions.numpy())
y_test_original_scale = scaler_y.inverse_transform(y_test_tf.numpy())
