import sys

sys.path.insert(0, "training")
from scripts.train import train_single
from scripts.load_data import load_dataset
from sklearn.model_selection import train_test_split

X, y, classes = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

params = {
    "epochs": 1,
    "batch_size": 64,
    "lr": 0.001,
    "optimizer": "Adam",
    "aug_factor": 1,
    "noise_std": 0.005,
    "dropout": 0.3,
    "num_points": 4096,
    "num_classes": 3,
}

print("Training quick model with params:", params)
model, metrics, run_id = train_single(params, X_train, y_train, X_test, y_test)
print(f"Run ID: {run_id}")
print(f"Metrics: {metrics}")
print("Model saved as", f"training/data/models/{run_id}.keras")
