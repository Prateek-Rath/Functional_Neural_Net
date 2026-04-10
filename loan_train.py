import csv
import time
from neural_network import Model, Trainer

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

if __name__ == "__main__":
    print("--- Modular Loan Data Training (Python) ---")
    limit = 10000
    x_data, y_data = load_loan_data("loan_data.csv", limit)
    print(f"Loaded {len(x_data)} samples.")

    # Architecture: 13 -> 16 -> 2
    model = Model([13, 16, 2])
    
    lr = 0.1
    epochs = 20
    batch_size = 32
    
    t_start = time.time()
    model = Trainer.train(model, x_data, y_data, epochs, batch_size, lr)
    t_end = time.time()
    
    # Calculate final accuracy on the same set for comparison
    final_acts, _ = model.forward(x_data)
    correct = 0
    for i in range(len(y_data)):
        pred = final_acts[-1][i].index(max(final_acts[-1][i]))
        if pred == y_data[i]:
            correct += 1
    
    accuracy = (correct / len(y_data)) * 100.0
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Total time: {t_end - t_start:.4f} seconds")
    print(f"RESULT: {accuracy:.2f}, {t_end - t_start:.4f}, {epochs}")
