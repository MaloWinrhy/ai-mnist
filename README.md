# Multi-Layer Perceptron (MLP) – Iris Dataset Classifier

This project implements a simple multi-layer perceptron (MLP) in Rust for classifying Iris flowers using supervised learning.

## Features

- Pure Rust implementation (no external ML frameworks)
- Multi-class classification (Setosa, Versicolor, Virginica)
- One-hot label encoding
- Softmax activation on output
- Cross-entropy loss
- Per-class accuracy and confusion matrix
- Train/test data split
- CSV data export for visualization

## Project Structure

- `src/mlp.rs`: Core MLP model
- `src/train.rs`: Training and evaluation logic
- `src/iris_loader.rs`: Load and preprocess Iris dataset
- `src/split.rs`: Train/test split utility
- `iris.csv`: Raw Iris dataset
- `iris_predictions.csv`: Exported predictions
- `iris_confusion_matrix.png`: Result visualization
- `iris_visualisation.py`: Optional Python plotting script

## How to Run

1. Make sure you have Rust installed: https://rustup.rs
2. Clone the repository and navigate to the project folder
3. Run:

```bash
cargo run --release
```

## Requirements

* Rust stable
* CSV file: `iris.csv` (included)

## License
# ai-mnist
