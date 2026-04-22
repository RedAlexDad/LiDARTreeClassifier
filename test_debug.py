import sys

sys.path.insert(0, "training")
from scripts.train import train_single
from scripts.load_data import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

print("Loading dataset...")
X, y, classes = load_dataset()
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

params = {
    "epochs": 2,
    "batch_size": 16,
    "lr": 0.001,
    "optimizer": "Adam",
    "aug_factor": 2,
    "noise_std": 0.005,
    "dropout": 0.3,
    "num_points": 4096,
    "num_classes": 3,
}

print("Starting train_single...")
model, metrics, run_id = train_single(params, X_train, y_train, X_test, y_test)
print(f"Run ID: {run_id}")
print(f"Metrics: {metrics}")
