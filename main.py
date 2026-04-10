import numpy as np


# ---------------- ACTIVATIONS ---------------- #
def relu(x):
    # x: (batch_size, n)
    return np.maximum(0, x)

def relu_derivative(x):
    # x: (batch_size, n)
    return (x > 0).astype(float)

def softmax(x):
    # x: (batch_size, num_classes)
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # (batch_size, num_classes)
    return exp / np.sum(exp, axis=1, keepdims=True)     # (batch_size, num_classes)


# ---------------- NEURAL NETWORK ---------------- #
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=(128, 64), output_size=10):
        self.weights = []
        self.biases = []

        # W0: (784, 128)
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        # b0: (1, 128)
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            # Wi: (128, 64)
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            # bi: (1, 64)
            self.biases.append(np.zeros((1, hidden_layers[i+1])))

        # W_last: (64, 10)
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        # b_last: (1, 10)
        self.biases.append(np.zeros((1, output_size)))


    # -------- FORWARD PASS -------- #
    def forward(self, X):
        # X: (batch_size, 784)
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            # (batch_size, prev_layer) dot (prev_layer, next_layer)
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            # z: (batch_size, next_layer)
            self.z_values.append(z)

            if i == len(self.weights) - 1:
                a = softmax(z)  # (batch_size, 10)
            else:
                a = relu(z)     # (batch_size, hidden_units)

            self.activations.append(a)

        return self.activations[-1]  # (batch_size, 10)


    # -------- LOSS -------- #
    def compute_loss(self, y_pred, y_true):
        # y_pred: (batch_size, 10)
        # y_true: (batch_size,)
        m = y_true.shape[0]

        log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)  # (batch_size,)
        return np.mean(log_likelihood)  # scalar


    # -------- BACKPROP -------- #
    def backward(self, y_true, learning_rate=0.01):
        m = y_true.shape[0]

        grads_w = []
        grads_b = []

        # dz: (batch_size, 10)
        dz = self.activations[-1].copy()
        dz[range(m), y_true] -= 1
        dz /= m

        for i in reversed(range(len(self.weights))):
            # activations[i]: (batch_size, prev_layer)
            # dz: (batch_size, current_layer)

            # dw: (prev_layer, current_layer)
            dw = np.dot(self.activations[i].T, dz)

            # db: (1, current_layer)
            db = np.sum(dz, axis=0, keepdims=True)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i != 0:
                # da: (batch_size, prev_layer)
                da = np.dot(dz, self.weights[i].T)

                # dz: (batch_size, prev_layer)
                dz = da * relu_derivative(self.z_values[i-1])

        # Update
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]


    # -------- TRAINING LOOP -------- #
    def train(self, X, y, epochs=10, learning_rate=0.01):
        # X: (batch_size, 784)
        for epoch in range(epochs):
            y_pred = self.forward(X)  # (batch_size, 10)
            loss = self.compute_loss(y_pred, y)
            self.backward(y, learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def train_batch(self, X, y, epochs=10, batch_size=32, lr=0.01):
    n = X.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = self.forward(X_batch)
            self.backward(y_batch, lr)

        loss = self.compute_loss(self.forward(X), y)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # -------- PREDICTION -------- #
    def predict(self, X):
        # returns: (batch_size,)
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


    # -------- ACCURACY -------- #
    def accuracy(self, X, y):
        preds = self.predict(X)  # (batch_size,)
        return np.mean(preds == y)


# ---------------- EXAMPLE ---------------- #
if __name__ == "__main__":
    np.random.seed(42)

    # X: (1000, 784)
    X = np.random.randn(1000, 784)

    # y: (1000,)
    y = np.random.randint(0, 10, size=1000)

    nn = NeuralNetwork()

    nn.train(X, y, epochs=10, learning_rate=0.01)

    print("Accuracy:", nn.accuracy(X, y))
