import math
import random

class Matrix:
    @staticmethod
    def matmul(A, B):
        m, n = len(A), len(A[0])
        p = len(B[0])
        C = [[0] * p for _ in range(m)]
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @staticmethod
    def add_bias(A, b):
        return [[rav + rbv for rav, rbv in zip(ra, b[0])] for ra in A]

    @staticmethod
    def transpose(A):
        return list(map(list, zip(*A)))

    @staticmethod
    def sum_axis_0(A):
        m, n = len(A), len(A[0])
        sums = [[0.0] * n]
        for i in range(m):
            for j in range(n):
                sums[0][j] += A[i][j]
        return sums

class NeuralNetwork:
    @staticmethod
    def relu(x):
        return [[max(0, v) for v in row] for row in x]

    @staticmethod
    def relu_derivative(z):
        return [[(1 if v > 0 else 0) for v in row] for row in z]

    @staticmethod
    def softmax(x):
        res = []
        for row in x:
            max_val = max(row)
            exps = [math.exp(v - max_val) for v in row]
            sum_exps = sum(exps) + 1e-9
            res.append([v / sum_exps for v in exps])
        return res

    @staticmethod
    def compute_loss(a_last, y_true):
        m = len(y_true)
        loss = 0
        for i in range(m):
            loss -= math.log(a_last[i][y_true[i]] + 1e-9)
        return loss / m

class Model:
    def __init__(self, layers_dims, seed=42):
        random.seed(seed)
        self.layers = []
        for i in range(len(layers_dims) - 1):
            fan_in = layers_dims[i]
            fan_out = layers_dims[i+1]
            scale = 0.01
            w = [[(random.random() * 2 - 1) * scale for _ in range(fan_out)] for _ in range(fan_in)]
            b = [[0.0] * fan_out]
            self.layers.append({'w': w, 'b': b})

    def forward(self, x):
        activations = [x]
        zs = []
        a = x
        for i, layer in enumerate(self.layers):
            z = Matrix.add_bias(Matrix.matmul(a, layer['w']), layer['b'])
            zs.append(z)
            if i == len(self.layers) - 1:
                a = NeuralNetwork.softmax(z)
            else:
                a = NeuralNetwork.relu(z)
            activations.append(a)
        return activations, zs

    def backward(self, activations, zs, y_true):
        m = len(y_true)
        a_last = activations[-1]
        
        # Initial dz for softmax + cross entropy
        dz = [[(a_last[i][j] - (1 if j == y_true[i] else 0)) / m for j in range(len(a_last[0]))] for i in range(m)]
        
        grads = []
        for i in reversed(range(len(self.layers))):
            a_prev = activations[i]
            layer = self.layers[i]
            
            dw = Matrix.matmul(Matrix.transpose(a_prev), dz)
            db = Matrix.sum_axis_0(dz)
            grads.insert(0, (dw, db))
            
            if i > 0:
                da = Matrix.matmul(dz, Matrix.transpose(layer['w']))
                z_prev = zs[i-1]
                dz = [[da[row][col] * (1 if z_prev[row][col] > 0 else 0) for col in range(len(z_prev[0]))] for row in range(len(z_prev))]
                
        return grads

    def update(self, grads, lr):
        for i in range(len(self.layers)):
            dw, db = grads[i]
            for j in range(len(self.layers[i]['w'])):
                for k in range(len(self.layers[i]['w'][0])):
                    self.layers[i]['w'][j][k] -= lr * dw[j][k]
            for k in range(len(self.layers[i]['b'][0])):
                self.layers[i]['b'][0][k] -= lr * db[0][k]

class Trainer:
    @staticmethod
    def train(model, x_data, y_data, epochs, batch_size, lr):
        m = len(x_data)
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for i in range(0, m, batch_size):
                x_batch = x_data[i : i+batch_size]
                y_batch = y_data[i : i+batch_size]
                
                acts, zs = model.forward(x_batch)
                loss = NeuralNetwork.compute_loss(acts[-1], y_batch)
                epoch_loss += loss * len(x_batch)
                
                # Check accuracy
                for b_idx in range(len(y_batch)):
                    pred = acts[-1][b_idx].index(max(acts[-1][b_idx]))
                    if pred == y_batch[b_idx]:
                        correct += 1
                
                grads = model.backward(acts, zs, y_batch)
                model.update(grads, lr)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss/m:.4f}, Acc: {correct/m*100:.2f}%")
        return model
