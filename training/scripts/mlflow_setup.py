import mlflow
import mlflow.keras
import os

MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "../data/mlruns")


def setup_mlflow():
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_DIR)}")
    mlflow.set_experiment("LiDAR-Tree-PointNet")
    return mlflow