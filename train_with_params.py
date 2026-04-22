import os
import sys
import argparse

sys.path.insert(0, "training")

import mlflow
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from models.pointnet import build_pointnet
from scripts.load_data import load_dataset
from scripts.mlflow_setup import setup_mlflow


def train_with_params(
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str = "Adam",
    dropout: float = 0.3,
    test_size: float = 0.2,
):
    print(f"[INFO] Loading dataset...")
    X, y, classes = load_dataset()
    print(f"[INFO] Dataset: X={X.shape}, y={y.shape}, classes={classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    os.makedirs("training/data/models", exist_ok=True)
    setup_mlflow()

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": optimizer,
        "dropout": dropout,
        "num_points": 4096,
        "num_classes": 3,
    }

    run_name = f"ep{epochs}_bs{batch_size}_lr{lr}_{optimizer}_drop{dropout}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[INFO] Run ID: {run_id}")

        mlflow.log_params(params)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(len(X_train)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(batch_size)

        print(f"[INFO] Building model...")
        model = build_pointnet(
            num_points=4096,
            num_classes=3,
            dropout_rate=dropout,
        )

        if optimizer == "Adam":
            opt = keras.optimizers.Adam(learning_rate=lr)
        else:
            opt = keras.optimizers.SGD(learning_rate=lr)

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        print(f"[INFO] Training: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1,
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics(
                {
                    "epoch": epoch + 1,
                    "train_loss": history.history["loss"][epoch],
                    "train_acc": history.history["sparse_categorical_accuracy"][epoch],
                    "val_loss": history.history["val_loss"][epoch],
                    "val_acc": history.history["val_sparse_categorical_accuracy"][
                        epoch
                    ],
                },
                step=epoch + 1,
            )

        final_metrics = {
            "final_train_acc": history.history["sparse_categorical_accuracy"][-1],
            "final_val_acc": history.history["val_sparse_categorical_accuracy"][-1],
            "best_val_acc": max(history.history["val_sparse_categorical_accuracy"]),
            "best_epoch": history.history["val_sparse_categorical_accuracy"].index(
                max(history.history["val_sparse_categorical_accuracy"])
            )
            + 1,
        }
        mlflow.log_metrics(final_metrics)

        preds = model.predict(test_ds, verbose=0)
        preds = tf.math.argmax(preds, -1).numpy()
        cm = confusion_matrix(y_test, preds)
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist()}, artifact_file="confusion_matrix.json"
        )

        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            mlflow.log_metrics(
                {f"precision_class_{i}": precision, f"recall_class_{i}": recall}
            )

        model_path = f"training/data/models/{run_id}.keras"
        model.save(model_path)
        mlflow.keras.log_model(model, artifact_path="model")

        print(f"\n[RESULTS] {run_name}")
        print(f"  Final Train Accuracy: {final_metrics['final_train_acc']:.4f}")
        print(f"  Final Val Accuracy: {final_metrics['final_val_acc']:.4f}")
        print(
            f"  Best Val Accuracy: {final_metrics['best_val_acc']:.4f} (epoch {final_metrics['best_epoch']})"
        )
        print(f"  Confusion Matrix:\n{cm}")
        print(f"  Model saved: {model_path}")

        return model, final_metrics, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PointNet with custom hyperparameters"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer (Adam/SGD)"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size ratio")

    args = parser.parse_args()

    train_with_params(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        dropout=args.dropout,
        test_size=args.test_size,
    )
