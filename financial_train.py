import csv
import time
from neural_network import Model, Trainer

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

if __name__ == "__main__":
    print("--- Modular Financial Data Training (Python) ---")
    dataset_path = "simulated_financial_forecasting_data.csv"
    x_data, y_data = load_financial_data(dataset_path)
    print(f"Loaded {len(x_data)} samples.")

    # Architecture: 6 inputs -> 8 hidden -> 2 output classes
    model = Model([6, 8, 2])
    
    epochs = 20
    batch_size = 16
    lr = 0.5
    
    t_start = time.time()
    model = Trainer.train(model, x_data, y_data, epochs, batch_size, lr)
    t_end = time.time()
    
    print(f"Total Training Time: {t_end - t_start:.4f} seconds")
    # Accuracy from the last epoch or calculated here for modular trainer
    final_acts, _ = model.forward(x_data)
    correct = 0
    for i in range(len(y_data)):
        pred = final_acts[-1][i].index(max(final_acts[-1][i]))
        if pred == y_data[i]:
            correct += 1
    accuracy = (correct / len(y_data)) * 100.0
    print(f"RESULT: {accuracy:.2f}, {t_end - t_start:.4f}, {epochs}")
