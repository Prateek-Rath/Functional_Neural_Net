import subprocess
import re
import csv
import os

def run_command(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

def extract_result(output):
    match = re.search(r"RESULT:\s*([\d.]+),\s*([\d.]+),\s*(\d+)", output)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "N/A", "N/A", "N/A"

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_file = os.path.join(results_dir, "benchmark_results.csv")
    
    # 1. Compile OCaml
    print("Compiling OCaml binaries...")
    run_command("/opt/homebrew/bin/ocamlc unix.cma -o financial_train_bench matrix.ml nn.ml train.ml financial_loader.ml financial_train.ml")
    run_command("/opt/homebrew/bin/ocamlc unix.cma -o loan_train_bench matrix.ml nn.ml train.ml loan_loader.ml loan_train.ml")

    benchmarks = [
        {"name": "Financial Data", "lang": "OCaml", "cmd": "./financial_train_bench"},
        {"name": "Financial Data", "lang": "Python", "cmd": "python3 financial_train.py"},
        {"name": "Loan Data", "lang": "OCaml", "cmd": "./loan_train_bench"},
        {"name": "Loan Data", "lang": "Python", "cmd": "python3 loan_train.py"},
    ]

    all_results = []
    for bench in benchmarks:
        print(f"Running {bench['name']} ({bench['lang']})...")
        stdout, _ = run_command(bench['cmd'])
        acc, time_taken, epochs = extract_result(stdout)
        all_results.append({
            "Dataset": bench['name'],
            "Language": bench['lang'],
            "Accuracy (%)": acc,
            "Time (s)": time_taken,
            "Epochs": epochs
        })

    # Write to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Language", "Accuracy (%)", "Time (s)", "Epochs"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nBenchmarking complete! Results saved to {csv_file}")
    
    # Print the table for immediate feedback
    print("\n--- Summary Table ---")
    print(f"{'Dataset':<20} | {'Lang':<8} | {'Accuracy':<10} | {'Time (s)':<10} | {'Epochs':<8}")
    print("-" * 65)
    for res in all_results:
        print(f"{res['Dataset']:<20} | {res['Language']:<8} | {res['Accuracy (%)']:<10} | {res['Time (s)']:<10} | {res['Epochs']:<8}")

if __name__ == "__main__":
    main()
