import os
import sys

sys.path.insert(0, "training")

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from models.pointnet import OrthogonalRegularizer
from scripts.load_data import load_dataset


CLASS_MAP = {0: "Береза", 1: "Ель", 2: "Сосна"}


def visualize_predictions(model_path: str, num_samples: int = 8):
    model = keras.models.load_model(
        model_path, custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer}
    )

    X, Y, _ = load_dataset()
    indices = np.random.choice(len(X), num_samples, replace=False)

    fig = plt.figure(figsize=(15, 15))
    points_list = []
    labels_list = []
    preds_list = []

    for idx in indices:
        points = X[idx]
        label = Y[idx]
        points_list.append(points)
        labels_list.append(label)

        pred = model.predict(points[tf.newaxis, ...], verbose=0)
        pred = tf.math.argmax(pred, -1).numpy()[0]
        preds_list.append(pred)

    points_arr = np.array(points_list)

    for i in range(num_samples):
        ax = fig.add_subplot(4, 2, i + 1, projection="3d")
        ax.scatter(points_arr[i, :, 0], points_arr[i, :, 1], points_arr[i, :, 2], s=5)
        ax.set_title(
            f"pred: {CLASS_MAP[preds_list[i]]}, label: {CLASS_MAP[labels_list[i]]}"
        )
        ax.set_axis_off()

    plt.tight_layout()
    save_path = model_path.replace(".keras", "_3d_predictions.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples")
    args = parser.parse_args()

    visualize_predictions(args.model, args.samples)
