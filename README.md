# Functional Neural Network from Scratch

A comparative implementation of a Multi-Layer Perceptron (MLP) built from scratch in both **OCaml** and **Python**. This project demonstrates the feasibility and efficiency of building complex machine learning models using a strictly functional programming paradigm (no iterative loops, no mutation).

## Project Overview

- **OCaml Implementation**: Strictly functional. Uses tail-recursion, `List.map`, `List.fold`, and immutable state passing for the entire training process.
- **Python Implementation**: Modular refactor of typical iterative logic, mirroring the OCaml architecture for a direct 1-to-1 comparison.
- **Goal**: Compare raw performance, code readability, and mathematical parity between functional and iterative approaches.

## Features

- **Matrix Library**: Pure functional matrix operations (multiply, transpose, etc.).
- **Generic MLP**: Supports arbitrary layer counts and dimensions.
- **Activation Functions**: ReLU (hidden layers) and Softmax (output layer).
- **Optimizer**: Mini-batch Stochastic Gradient Descent (SGD) with Cross-Entropy Loss.
- **Datasets**: Built-in loaders for XOR, Synthetic Financial Forecasting, and Loan Status prediction.

## Directory Structure

| File | Description |
| :--- | :--- |
| `matrix.ml` | OCaml core matrix math using lists. |
| `nn.ml` | OCaml Forward/Backward pass and loss calculation. |
| `train.ml` | OCaml high-level training loop and batching. |
| `neural_network.py` | Python modular core (Mirrors the OCaml logic). |
| `benchmark.py` | Automated suite to run, compare, and record results. |
| `results/` | Auto-generated CSV files comparing performance. |

## Getting Started

### Prerequisites

- **OCaml**: `ocamlc` or `ocamlopt` (part of the standard OCaml distribution).
- **Python**: version 3.x.
- **Unix Environment**: (Mac/Linux) for benchmarking script support.

### How to Run Benchmarks

To run the full comparative suite across all datasets (compiles OCaml automatically):

```bash
python3 benchmark.py
```

Results will be printed as a table and saved to `results/benchmark_results.csv`.

### Manual Execution

**OCaml Training (Loan Data):**
```bash
/opt/homebrew/bin/ocamlc unix.cma -o loan_train matrix.ml nn.ml train.ml loan_loader.ml loan_train.ml
./loan_train
```

**Python Training (Loan Data):**
```bash
python3 loan_train.py
```

## Performance & Accuracy

As of the latest run, both implementations achieve parity in accuracy while OCaml demonstrates impressive efficiency for a functional implementation using linked lists.

| Dataset | Language | Accuracy (%) | Time |
| :--- | :--- | :--- | :--- |
| Financial Forecasting | OCaml | ~94% | ~0.3s |
| Loan Status Approval | OCaml | ~85% | ~8.5s |

## Authors
- **Ketan Raman Ghungralekar**
- **Prateek Rath**
