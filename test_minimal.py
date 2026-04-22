import sys

sys.path.insert(0, "training")
from models.pointnet import build_pointnet
import tensorflow as tf
import numpy as np

print("Building model...")
model = build_pointnet(num_points=4096, num_classes=3, dropout_rate=0.3)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
print("Model built")

# Create dummy data
X_dummy = np.random.randn(10, 4096, 3).astype(np.float32)
y_dummy = np.random.randint(0, 3, size=10).astype(np.int32)
print("Dummy data created")

# Train for 1 epoch
history = model.fit(X_dummy, y_dummy, epochs=1, verbose=1)
print("Training completed")
print(f"Loss: {history.history['loss'][0]}, Accuracy: {history.history['accuracy'][0]}")
