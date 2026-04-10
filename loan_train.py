import csv
import math
import time
import random

def relu(x):
    return [max(0, v) for v in x]

def softmax(x):
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    sum_exps = sum(exps) + 1e-9
    return [v / sum_exps for v in exps]

def matmul(A, B):
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def add_bias(A, b):
    return [[rav + rbv for rav, rbv in zip(ra, b[0])] for ra in A]

# Label Encoders Matching OCaml
def gender_map(s): return 0.0 if s == "female" else 1.0
def edu_map(s):
    return {"Associate": 0.0, "Bachelor": 1.0, "Doctorate": 2.0, "High School": 3.0, "Master": 4.0}.get(s, 0.0)
def home_map(s):
    return {"MORTGAGE": 0.0, "OTHER": 1.0, "OWN": 2.0, "RENT": 3.0}.get(s, 0.0)
def intent_map(s):
    return {"DEBTCONSOLIDATION": 0.0, "EDUCATION": 1.0, "HOMEIMPROVEMENT": 2.0, "MEDICAL": 3.0, "PERSONAL": 4.0, "VENTURE": 5.0}.get(s, 0.0)
def default_map(s): return 0.0 if s == "No" else 1.0

def load_loan_data(filename, limit=10000):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit: break
            # Indices match OCaml parse_row
            age = float(row[0])
            gender = gender_map(row[1])
            edu = edu_map(row[2])
            income = float(row[3])
            exp = float(row[4])
            home = home_map(row[5])
            amnt = float(row[6])
            intent = intent_map(row[7])
            rate = float(row[8]) if row[8] else 0.0
            pct = float(row[9])
            hist = float(row[10])
            score = float(row[11])
            dflt = default_map(row[12])
            label = int(row[13])
            
            rows.append(([age, gender, edu, income, exp, home, amnt, intent, rate, pct, hist, score, dflt], label))
    
    x_raw = [r[0] for r in rows]
    y = [r[1] for r in rows]
    
    # Normalize
    num_cols = len(x_raw[0])
    stats = []
    for j in range(num_cols):
        col = [row[j] for row in x_raw]
        stats.append((min(col), max(col)))
    
    x = []
    for row in x_raw:
        new_row = []
        for j in range(num_cols):
            min_v, max_v = stats[j]
            if max_v - min_v < 1e-9: new_row.append(0.0)
            else: new_row.append((row[j] - min_v) / (max_v - min_v))
        x.append(new_row)
    
    return x, y

def init_layer(fan_in, fan_out):
    scale = 0.01
    random.seed(42)
    w = [[(random.random() * 2 - 1) * scale for _ in range(fan_out)] for _ in range(fan_in)]
    b = [[0.0] * fan_out]
    return {'w': w, 'b': b}

if __name__ == "__main__":
    print("--- Loan Data Training (Python) ---")
    limit = 10000
    x_data, y_data = load_loan_data("loan_data.csv", limit)
    print(f"Loaded {len(x_data)} samples.")

    # Architecture: 13 -> 16 -> 2
    l1 = init_layer(13, 16)
    l2 = init_layer(16, 2)
    
    lr = 0.1
    epochs = 20
    batch_size = 32
    
    t_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        # Mini-batch loop
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]
            m_batch = len(x_batch)
            
            # Forward
            z1 = add_bias(matmul(x_batch, l1['w']), l1['b'])
            a1 = [[max(0, v) for v in row] for row in z1]
            z2 = add_bias(matmul(a1, l2['w']), l2['b'])
            a2 = [softmax(row) for row in z2]
            
            # Loss & Stats
            for b_idx in range(m_batch):
                epoch_loss -= math.log(a2[b_idx][y_batch[b_idx]] + 1e-9)
                if a2[b_idx].index(max(a2[b_idx])) == y_batch[b_idx]:
                    correct += 1
            
            # Backward
            dz2 = [[(a2[b_idx][j] - (1 if j == y_batch[b_idx] else 0)) / m_batch for j in range(2)] for b_idx in range(m_batch)]
            dw2 = matmul(list(map(list, zip(*a1))), dz2)
            db2 = [[sum(dz2[b_idx][j] for b_idx in range(m_batch)) for j in range(2)]]
            
            da1 = matmul(dz2, list(map(list, zip(*l2['w']))))
            dz1 = [[da1[b_idx][j] * (1 if z1[b_idx][j] > 0 else 0) for j in range(16)] for b_idx in range(m_batch)]
            dw1 = matmul(list(map(list, zip(*x_batch))), dz1)
            db1 = [[sum(dz1[b_idx][j] for b_idx in range(m_batch)) for j in range(16)]]
            
            # Update (Mini-batch step)
            for j in range(len(l1['w'])):
                for k in range(len(l1['w'][0])): l1['w'][j][k] -= lr * dw1[j][k]
            for k in range(len(l1['b'][0])): l1['b'][0][k] -= lr * db1[0][k]
            for j in range(len(l2['w'])):
                for k in range(len(l2['w'][0])): l2['w'][j][k] -= lr * dw2[j][k]
            for k in range(len(l2['b'][0])): l2['b'][0][k] -= lr * db2[0][k]

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(y_data):.4f}, Acc: {correct/len(y_data)*100:.2f}%")
        
    t_end = time.time()
    print(f"Final Accuracy: {correct/len(y_data)*100:.2f}%")
    print(f"Total time: {t_end - t_start:.4f} seconds")
