import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-9)

# XOR Data
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y = np.array([0, 1, 1, 0])

# Fixed Weights (2 -> 4 -> 2)
w1 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8]
])
b1 = np.array([[0.01, 0.02, 0.03, 0.04]])
w2 = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])
b2 = np.array([[0.05, 0.06]])

weights = [w1, w2]
biases = [b1, b2]

# Forward
activations = [X]
z_values = []

# Layer 1
z1 = np.dot(activations[-1], weights[0]) + biases[0]
z_values.append(z1)
a1 = relu(z1)
activations.append(a1)

# Layer 2
z2 = np.dot(activations[-1], weights[1]) + biases[1]
z_values.append(z2)
a2 = softmax(z2)
activations.append(a2)

# Loss
m = y.shape[0]
y_pred = activations[-1]
log_likelihood = -np.log(y_pred[range(m), y] + 1e-9)
loss = np.mean(log_likelihood)
print(f"Initial loss: {loss:.6f}")

# Backward
lr = 0.5
dz = activations[-1].copy()
dz[range(m), y] -= 1
dz /= m

# Layer 2 grads
dw2 = np.dot(activations[1].T, dz)
db2 = np.sum(dz, axis=0, keepdims=True)

# Layer 1 grads
da1 = np.dot(dz, weights[1].T)
dz1 = da1 * (z1 > 0).astype(float)
dw1 = np.dot(activations[0].T, dz1)
db1 = np.sum(dz1, axis=0, keepdims=True)

# Update
weights[1] -= lr * dw2
biases[1] -= lr * db2
weights[0] -= lr * dw1
biases[0] -= lr * db1

# Forward again
activations = [X]
z1 = np.dot(activations[-1], weights[0]) + biases[0]
a1 = relu(z1)
activations.append(a1)
z2 = np.dot(activations[-1], weights[1]) + biases[1]
a2 = softmax(z2)
activations.append(a2)

loss2 = np.mean(-np.log(activations[-1][range(m), y] + 1e-9))
print(f"Loss after 1 step: {loss2:.6f}")
