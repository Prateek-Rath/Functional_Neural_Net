import csv
import math
import time

def relu(x):
    return [max(0, v) for v in x]

def softmax(x):
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    sum_exps = sum(exps) + 1e-9
    return [v / sum_exps for v in exps]

def matmul(A, B):
    # A: MxN, B: NxP -> MxP
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def add_bias(A, b):
    # A: MxN, b: [[1xN]]
    return [[rav + rbv for rav, rbv in zip(ra, b[0])] for ra in A]

def load_financial_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [[float(v) for v in row] for row in reader]
    
    x_raw = [row[:6] for row in data]
    y_raw = [row[6] for row in data]
    
    mean_y = sum(y_raw) / len(y_raw)
    y = [0 if v < mean_y else 1 for v in y_raw]
    
    # Normalize X
    num_cols = len(x_raw[0])
    stats = []
    for j in range(num_cols):
        col_vals = [row[j] for row in x_raw]
        stats.append((min(col_vals), max(col_vals)))
    
    x = []
    for row in x_raw:
        new_row = []
        for j in range(num_cols):
            min_v, max_v = stats[j]
            if max_v - min_v < 1e-9:
                new_row.append(0.0)
            else:
                new_row.append((row[j] - min_v) / (max_v - min_v))
        x.append(new_row)
        
    return x, y

def init_layer(fan_in, fan_out):
    scale = 0.01
    import random
    random.seed(42)
    w = [[(random.random() * 2 - 1) * scale for _ in range(fan_out)] for _ in range(fan_in)]
    b = [[0.0] * fan_out]
    return {'w': w, 'b': b}

if __name__ == "__main__":
    print("--- Financial Data Training (Python) ---")
    x_data, y_data = load_financial_data("simulated_financial_forecasting_data.csv")
    print(f"Loaded {len(x_data)} samples.")

    # Architecture: 6 -> 8 -> 2
    l1 = init_layer(6, 8)
    l2 = init_layer(8, 2)
    
    # Minimal training loop to match OCaml's "from scratch" feel
    # (Since I can't use numpy here easily, I'll keep it simple/functional)
    
    # Actually, to truly compare, I'll use the same learning rate and logic.
    # Note: Pure Python list-based matmul is much slower than C-based OCaml.
    
    t_start = time.time()
    # We will just run the forward pass and a few epochs to show it working.
    print("Starting training (Python)...")
    
    # For speed in pure python (no numpy), let's just do 5 epochs
    lr = 0.5
    epochs = 5
    for epoch in range(epochs):
        # Forward
        z1 = add_bias(matmul(x_data, l1['w']), l1['b'])
        a1 = [[max(0, v) for v in row] for row in z1]
        z2 = add_bias(matmul(a1, l2['w']), l2['b'])
        a2 = [softmax(row) for row in z2]
        
        # Loss
        loss = 0
        for i in range(len(y_data)):
            loss -= math.log(a2[i][y_data[i]] + 1e-9)
        loss /= len(y_data)
        
        # Backward (Simple Batch)
        m = len(y_data)
        dz2 = [[(a2[i][j] - (1 if j == y_data[i] else 0)) / m for j in range(2)] for i in range(m)]
        
        # dw2 = a1.T * dz2
        dw2 = [[0]*2 for _ in range(8)]
        for i in range(m):
            for j in range(8):
                for k in range(2):
                    dw2[j][k] += a1[i][j] * dz2[i][k]
        
        db2 = [[sum(dz2[i][j] for i in range(m)) for j in range(2)]]
        
        # da1 = dz2 * l2_w.T
        da1 = [[sum(dz2[i][k] * l2['w'][j][k] for k in range(2)) for j in range(8)] for i in range(m)]
        dz1 = [[da1[i][j] * (1 if z1[i][j] > 0 else 0) for j in range(8)] for i in range(m)]
        
        # dw1 = x.T * dz1
        dw1 = [[0]*8 for _ in range(6)]
        for i in range(m):
            for j in range(6):
                for k in range(8):
                    dw1[j][k] += x_data[i][j] * dz1[i][k]
        
        db1 = [[sum(dz1[i][j] for i in range(m)) for j in range(8)]]
        
        # Update
        for j in range(6):
            for k in range(8):
                l1['w'][j][k] -= lr * dw1[j][k]
        for k in range(8):
            l1['b'][0][k] -= lr * db1[0][k]
        for j in range(8):
            for k in range(2):
                l2['w'][j][k] -= lr * dw2[j][k]
        for k in range(2):
            l2['b'][0][k] -= lr * db2[0][k]

        print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
        
    t_end = time.time()
    print(f"Total time (5 epochs): {t_end - t_start:.4f} seconds")
