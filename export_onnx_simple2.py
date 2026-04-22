import keras
import tf2onnx
import tensorflow as tf
import os
import sys

sys.path.insert(0, "training")
from models.pointnet import OrthogonalRegularizer

run_id = "4569272ae00946eca428eea3499dbe33"
keras_path = f"training/data/models/{run_id}.keras"
onnx_path = f"training/data/models/{run_id}.onnx"

print(f"Loading model from {keras_path}")
model = keras.models.load_model(
    keras_path, custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer}
)

print("Converting to ONNX...")
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    opset=13,
)

with open(onnx_path, "wb") as f:
    f.write(model_proto.SerializeToString())

size_kb = os.path.getsize(onnx_path) / 1024
print(f"ONNX saved: {onnx_path} ({size_kb:.1f} KB)")
