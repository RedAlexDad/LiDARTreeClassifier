import mlflow
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.pointnet import build_pointnet
from scripts.augmentation import augment_dataset
from scripts.load_data import load_dataset
from scripts.mlflow_setup import setup_mlflow


def train_single(
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    model_dir: str = "training/data/models",
):
    os.makedirs(model_dir, exist_ok=True)
    print("[DEBUG] Setting up MLflow...")
    setup_mlflow()
    print("[DEBUG] MLflow setup done")

    run_name = (
        f"ep{params['epochs']}_bs{params['batch_size']}_"
        f"lr{params['lr']}_{params['optimizer']}_"
        f"aug{params['aug_factor']}_drop{params['dropout']}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "learning_rate": params["lr"],
                "optimizer": params["optimizer"],
                "aug_factor": params["aug_factor"],
                "noise_std": params["noise_std"],
                "dropout": params["dropout"],
                "num_points": params.get("num_points", 4096),
                "num_classes": params.get("num_classes", 3),
            }
        )

        print(
            f"[DEBUG] Augmenting dataset: X_train shape {X_train.shape}, aug_factor={params['aug_factor']}"
        )
        X_aug, y_aug = augment_dataset(
            X_train,
            y_train,
            aug_factor=params["aug_factor"],
            noise_std=params["noise_std"],
        )
        print(f"[DEBUG] Augmented shape: {X_aug.shape}")
        mlflow.log_param("train_samples_original", len(X_train))
        mlflow.log_param("train_samples_augmented", len(X_aug))

        print("[DEBUG] Creating TensorFlow datasets...")
        train_ds = tf.data.Dataset.from_tensor_slices((X_aug, y_aug))
        print(f"[DEBUG] Train dataset created, shuffling {len(X_aug)} samples...")
        train_ds = train_ds.shuffle(min(100, len(X_aug))).batch(params["batch_size"])
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(params["batch_size"])
        print("[DEBUG] Datasets ready")

        print("[DEBUG] Building PointNet model...")
        model = build_pointnet(
            num_points=params.get("num_points", 4096),
            num_classes=params.get("num_classes", 3),
            dropout_rate=params["dropout"],
        )
        print("[DEBUG] Model built")

        print(
            f"[DEBUG] Creating optimizer: {params['optimizer']} with lr={params['lr']}"
        )
        if params["optimizer"] == "Adam":
            opt = keras.optimizers.Adam(learning_rate=params["lr"])
        else:
            opt = keras.optimizers.SGD(learning_rate=params["lr"])
        print("[DEBUG] Compiling model...")
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["sparse_categorical_accuracy"],
        )
        print("[DEBUG] Model compiled")

        print("[DEBUG] Setting up EarlyStopping callback")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        print("[DEBUG] Callbacks ready")

        print("[DEBUG] Starting model.fit...")
        history = model.fit(
            train_ds,
            epochs=params["epochs"],
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1,
        )
        print("[DEBUG] model.fit completed")

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
            "best_epoch": (
                history.history["val_sparse_categorical_accuracy"].index(
                    max(history.history["val_sparse_categorical_accuracy"])
                )
                + 1
            ),
            "overfit_gap": (
                history.history["loss"][-1] - history.history["val_loss"][-1]
            ),
        }
        mlflow.log_metrics(final_metrics)

        preds = model.predict(test_ds, verbose=0)
        preds = tf.math.argmax(preds, -1).numpy()
        cm = confusion_matrix(y_test, preds)
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist()}, artifact_file="confusion_matrix.json"
        )

        for i in range(params["num_classes"]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            mlflow.log_metrics(
                {
                    f"precision_class_{i}": precision,
                    f"recall_class_{i}": recall,
                }
            )

        keras_path = os.path.join(model_dir, f"{run_id}.keras")
        model.save(keras_path)
        mlflow.keras.log_model(model, artifact_path="model")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
            fig.add_trace(
                go.Scatter(
                    y=history.history["sparse_categorical_accuracy"],
                    name="train_acc",
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["val_sparse_categorical_accuracy"],
                    name="val_acc",
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["loss"], name="train_loss", mode="lines+markers"
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["val_loss"], name="val_loss", mode="lines+markers"
                ),
                row=1,
                col=2,
            )
            fig.update_layout(title_text=f"Training Run: {run_name}", showlegend=True)
            fig_path = os.path.join(model_dir, f"{run_id}_training_curve.html")
            fig.write_html(fig_path)
            mlflow.log_artifact(fig_path)
        except ImportError:
            pass

        return model, final_metrics, run_id
