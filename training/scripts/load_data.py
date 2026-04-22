import h5py
import numpy as np
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/v2.h5")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "../data/classes.json")


def load_dataset(data_path: str = None):
    if data_path is None:
        data_path = DATA_PATH

    h5f = h5py.File(data_path, "r")
    X = h5f.get("dataset_X")[:]
    Y_raw = h5f.get("dataset_Y")

    # Convert to string array if needed
    Y = Y_raw.asstr()[:] if Y_raw.dtype.kind == "S" else Y_raw[:]

    classes_map = json.loads(open(CLASSES_PATH).read())
    class_to_idx = {v: int(k) for k, v in classes_map.items()}
    # Ensure Y elements are strings (decode if bytes)
    Y = np.array(
        [class_to_idx[y.decode("utf-8") if isinstance(y, bytes) else y] for y in Y]
    )

    h5f.close()
    return X, Y, classes_map


if __name__ == "__main__":
    X, Y, classes = load_dataset()
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Classes: {classes}")
    print(f"Unique labels: {np.unique(Y)}")
    for k, v in classes.items():
        count = int(np.sum(Y == int(k)))
        print(f"  {k}: {v} -> {count} samples")
