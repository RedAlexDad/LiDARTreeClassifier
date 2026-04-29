import os
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train_single
from scripts.load_data import load_dataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aug-factor", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.005)
    args = parser.parse_args()

    X, y, classes = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "dropout": args.dropout,
        "aug_factor": args.aug_factor,
        "noise_std": args.noise_std,
        "num_points": 4096,
        "num_classes": 3,
    }

    print(f"Training with params: {params}")
    model, metrics, run_id = train_single(params, X_train, y_train, X_test, y_test)
    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")