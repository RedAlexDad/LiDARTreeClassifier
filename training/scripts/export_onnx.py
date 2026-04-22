import keras
import tf2onnx
import mlflow
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.mlflow_setup import setup_mlflow
from models.pointnet import OrthogonalRegularizer


def export_onnx(run_id: str, output_dir: str = "training/data/models"):
    os.makedirs(output_dir, exist_ok=True)
    # Load model directly from .keras file (bypass mlflow for simplicity)
    keras_path = os.path.join(output_dir, f"{run_id}.keras")
    if not os.path.exists(keras_path):
        # fallback to mlflow
        setup_mlflow()
        model_uri = f"runs:/{run_id}/model"
        keras_model = mlflow.keras.load_model(model_uri)
    else:
        # load with custom objects
        keras_model = keras.models.load_model(
            keras_path, custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer}
        )

    input_spec = [keras.Input(shape=(4096, 3), name="input_layer")]

    onnx_path = os.path.join(output_dir, f"{run_id}.onnx")

    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        opset_version=13,
    )

    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX saved: {onnx_path} ({size_kb:.1f} KB)")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(onnx_path)

    return onnx_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    export_onnx(args.run_id)
