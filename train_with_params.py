import os
import sys
import argparse

sys.path.insert(0, "training")

import mlflow
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.pointnet import build_pointnet
from scripts.load_data import load_dataset
from scripts.mlflow_setup import setup_mlflow

try:
    mlflow.set_system_metrics(sampling_interval=1)
except AttributeError:
    pass


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    acc = history.history["sparse_categorical_accuracy"]
    val_acc = history.history["val_sparse_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    axes[0].plot(epochs, acc, "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, val_acc, "r-", label="Validation", linewidth=2)
    axes[0].set_title("Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss, "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, val_loss, "r-", label="Validation", linewidth=2)
    axes[1].set_title("Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Loss", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def export_to_onnx(model, save_path):
    input_spec = tf.TensorSpec(shape=(None, 4096, 3), dtype=tf.float32)
    try:
        model_for_export = tf.keras.Model(inputs=model.input, outputs=model.output)
        onnx_model = tf2onnx.convert.from_keras(
            model=model_for_export,
            input_signature=[input_spec],
            output_path=save_path,
            opset=18,
        )
        return save_path
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")
        return None


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
    class_names = list(classes.values())
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
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    run_name = f"ep{epochs}_bs{batch_size}_lr{lr}_{optimizer}_drop{dropout}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[INFO] Run ID: {run_id}")

        mlflow.log_params(params)
        mlflow.log_params({"classes": json.dumps(classes)})

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
        model.summary(print_fn=lambda x: print(f"[INFO] {x}"))

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
            keras.callbacks.TensorBoard(
                log_dir=f"training/data/models/{run_id}_tb",
                histogram_freq=0,
                write_graph=True,
            ),
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
                    "train_loss": float(history.history["loss"][epoch]),
                    "train_acc": float(
                        history.history["sparse_categorical_accuracy"][epoch]
                    ),
                    "val_loss": float(history.history["val_loss"][epoch]),
                    "val_acc": float(
                        history.history["val_sparse_categorical_accuracy"][epoch]
                    ),
                },
                step=epoch + 1,
            )

        final_metrics = {
            "final_train_acc": float(
                history.history["sparse_categorical_accuracy"][-1]
            ),
            "final_val_acc": float(
                history.history["val_sparse_categorical_accuracy"][-1]
            ),
            "best_val_acc": float(
                max(history.history["val_sparse_categorical_accuracy"])
            ),
            "best_epoch": int(
                history.history["val_sparse_categorical_accuracy"].index(
                    max(history.history["val_sparse_categorical_accuracy"])
                )
                + 1
            ),
        }
        mlflow.log_metrics(final_metrics)

        preds = model.predict(test_ds, verbose=0)
        preds = tf.math.argmax(preds, -1).numpy()
        cm = confusion_matrix(y_test, preds)

        cm_path = f"training/data/models/{run_id}_confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(cm_path, "confusion_matrix")

        mlflow.log_dict(
            {"confusion_matrix": cm.tolist()}, artifact_file="confusion_matrix.json"
        )

        tc_path = f"training/data/models/{run_id}_training_curves.png"
        plot_training_curves(history, tc_path)
        mlflow.log_artifact(tc_path, "training_curves")

        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            mlflow.log_metrics(
                {
                    f"precision_class_{i}_{class_names[i]}": float(precision),
                    f"recall_class_{i}_{class_names[i]}": float(recall),
                    f"f1_class_{i}_{class_names[i]}": float(f1),
                }
            )

        report = classification_report(y_test, preds, target_names=class_names)
        report_path = f"training/data/models/{run_id}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "classification_report")

        model_path = f"training/data/models/{run_id}.keras"
        model.save(model_path)
        mlflow.keras.log_model(model, artifact_path="keras_model")

        try:
            import tf2onnx

            onnx_path = f"training/data/models/{run_id}.onnx"
            input_spec = tf.TensorSpec(shape=(None, 4096, 3), dtype=tf.float32)
            model_for_export = tf.keras.Model(inputs=model.input, outputs=model.output)
            tf2onnx.convert.from_keras(
                model=model_for_export,
                input_signature=[input_spec],
                output_path=onnx_path,
                opset=18,
            )
            mlflow.log_artifact(onnx_path, "onnx_model")
            print(f"[INFO] ONNX model saved: {onnx_path}")
        except ImportError:
            print("[WARN] tf2onnx not installed, skipping ONNX export")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

        results_df = pd.DataFrame(
            {
                "metric": [
                    "final_train_acc",
                    "final_val_acc",
                    "best_val_acc",
                    "best_epoch",
                ],
                "value": [
                    final_metrics["final_train_acc"],
                    final_metrics["final_val_acc"],
                    final_metrics["best_val_acc"],
                    final_metrics["best_epoch"],
                ],
            }
        )
        results_csv = f"training/data/models/{run_id}_results.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv, "results")

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
