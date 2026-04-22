import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.models.pointnet import build_pointnet


def test_model_shape():
    model = build_pointnet(4096, 3)
    dummy = np.random.randn(2, 4096, 3).astype(np.float32)
    pred = model(dummy, training=False)
    assert pred.shape == (2, 3)


def test_model_output_sum():
    model = build_pointnet(4096, 3)
    dummy = np.random.randn(1, 4096, 3).astype(np.float32)
    pred = model(dummy, training=False)
    assert np.isclose(pred.numpy().sum(), 1.0, atol=0.01)


if __name__ == "__main__":
    test_model_shape()
    test_model_output_sum()
    print("All tests passed!")
