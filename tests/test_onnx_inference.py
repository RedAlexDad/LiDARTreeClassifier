import numpy as np
import onnx
import onnxruntime as ort
import os
import pytest


def test_onnx_inference():
    onnx_path = "training/data/models/best.onnx"
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX file not found at {onnx_path}, skipping test")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 4096, 3).astype(np.float32)
    out = sess.run(None, {"input_layer": dummy})
    assert out[0].shape == (1, 3)
    assert np.isclose(out[0].sum(), 1.0, atol=0.01)


if __name__ == "__main__":
    test_onnx_inference()
    print("ONNX test passed!")
