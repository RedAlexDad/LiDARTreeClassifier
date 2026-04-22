# LiDARTreeClassifier: декомпозиционный план

## Описание проекта

Веб-система классификации пород деревьев (Береза, Ель, Сосна) по LiDAR-облакам точек
с использованием архитектуры PointNet. Система включает MLOps-пайплайн с трекингом
экспериментов через MLflow (локальное хранилище), конвертацию моделей в ONNX.

**Текущие результаты (базовая модель в notebook):**

- Train accuracy: ~71%
- Validation accuracy: ~54%
- Overfit gap: ~17%

---

## Структура проекта (итоговая)

```
LiDARTreeClassifier/
├── docs/
│   ├── decomposition_plan.md       # этот файл
│   └── mlflow_reference.md       # шпаргалка по MLflow API
│
├── training/
│   ├── configs/                   # конфиги экспериментов (YAML)
│   ├── models/                    # определения архитектур (PointNet)
│   ├── scripts/
│   │   ├── train.py             # основной скрипт обучения
│   │   ├── train_grid.py       # перебор гиперпараметров
│   │   ├── export_onnx.py      # конвертация в ONNX
│   │   └── evaluate.py          # evaluation и метрики
│   └── data/
│       ├── v2.h5                # данные
│       ├── classes.json          # маппинг классов
│       ├── models/              # сохранённые модели (.keras + .onnx)
│       └── mlruns/              # локальный MLflow трекинг
│
├── tests/
│   ├── test_model.py            # unit-тесты модели
│   └── test_inference.py        # тесты инференса
│
├── docker-compose.yml           # MLflow в Docker
├── Makefile                   # make train-* , make export-onnx
└── requirements.txt
```

---

## Этап 1 — Подготовка данных

- [ ] Создать директорию `training/data/`
- [ ] Скопировать `v2.h5` из `./content/` в `training/data/`
- [ ] Создать `training/data/classes.json`:

```json
{
  "0": "Береза",
  "1": "Ель",
  "2": "Сосна"
}
```

- [ ] Создать `training/scripts/load_data.py`:

```python
import h5py
import numpy as np
import json
import os

DATA_PATH = "training/data/v2.h5"
CLASSES_PATH = "training/data/classes.json"

def load_dataset():
    h5f = h5py.File(DATA_PATH, 'r')
    X = h5f['dataset_X'][:]
    Y_raw = h5f['dataset_Y'][:]

    if Y_raw.dtype.kind == 'S':
        Y = np.array([y.decode('utf-8') for y in Y_raw])
    else:
        Y = Y_raw

    classes_map = json.loads(open(CLASSES_PATH).read())
    class_to_idx = {v: int(k) for k, v in classes_map.items()}
    Y = np.array([class_to_idx[y] for y in Y])

    h5f.close()
    return X, Y, classes_map
```

**Чеклист:**

- [ ] Проверить `X.shape == (N, 4096, 3)`
- [ ] Проверить `len(set(Y)) == 3`
- [ ] Запустить и убедиться, что данные загружаются без ошибок

---

## Этап 2 — Аугментация данных

- [ ] Создать `training/scripts/augmentation.py`:

```python
import numpy as np

def augment_point_cloud(points: np.ndarray, noise_std: float = 0.005) -> np.ndarray:
    idx = np.random.choice(points.shape[0], size=points.shape[0], replace=True)
    augmented = points[idx].copy()
    augmented += np.random.normal(0, noise_std, augmented.shape)
    return augmented

def augment_dataset(X: np.ndarray, y: np.ndarray,
                   aug_factor: int = 4,
                   noise_std: float = 0.005):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        for _ in range(aug_factor):
            X_aug.append(augment_point_cloud(X[i], noise_std))
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)
```

**Чеклист:**

- [ ] Augmented shape совпадает с оригинальным
- [ ] Шум не выходит за разумные bounds (±3σ)

---

## Этап 3 — Архитектура PointNet

- [ ] Создать `training/models/pointnet.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def build_pointnet(num_points=4096, num_classes=3, dropout_rate=0.3):
    inputs = keras.Input(shape=(num_points, 3), name="input_layer")

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(dropout_rate)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
```

**Чеклист:**

- [ ] Модель компилируется без ошибок
- [ ] Summary: 748,076 параметров
- [ ] Модель сохраняется в `.keras`

---

## Этап 4 — MLflow: трекинг экспериментов

### 4.1 Настройка MLflow (локально)

- [ ] Установить: `pip install mlflow plotly`

- [ ] Создать `training/scripts/mlflow_setup.py`:

```python
import mlflow
import os

MLFLOW_DIR = "training/data/mlruns"

def setup_mlflow():
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_DIR)}")
    mlflow.set_experiment("LiDAR-Tree-PointNet")
    return mlflow
```

### 4.2 Логирование параметров

- [ ] Создать `training/scripts/train.py`:

```python
import mlflow
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.pointnet import build_pointnet
from scripts.augmentation import augment_dataset
from scripts.load_data import load_dataset
from scripts.mlflow_setup import setup_mlflow

def train_single(params: dict, X_train, y_train, X_test, y_test,
               model_dir: str = "training/data/models"):
    os.makedirs(model_dir, exist_ok=True)
    setup_mlflow()

    run_name = (f"ep{params['epochs']}_bs{params['batch_size']}_"
                f"lr{params['lr']}_{params['optimizer']}_"
                f"aug{params['aug_factor']}_drop{params['dropout']}")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["lr"],
            "optimizer": params["optimizer"],
            "aug_factor": params["aug_factor"],
            "noise_std": params["noise_std"],
            "dropout": params["dropout"],
            "num_points": params.get("num_points", 4096),
            "num_classes": params.get("num_classes", 3),
        })

        X_aug, y_aug = augment_dataset(
            X_train, y_train,
            aug_factor=params["aug_factor"],
            noise_std=params["noise_std"]
        )
        mlflow.log_param("train_samples_original", len(X_train))
        mlflow.log_param("train_samples_augmented", len(X_aug))

        train_ds = tf.data.Dataset.from_tensor_slices((X_aug, y_aug))
        train_ds = train_ds.shuffle(len(X_aug)).batch(params["batch_size"])
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(params["batch_size"])

        model = build_pointnet(
            num_points=params.get("num_points", 4096),
            num_classes=params.get("num_classes", 3),
            dropout_rate=params["dropout"]
        )

        if params["optimizer"] == "Adam":
            opt = keras.optimizers.Adam(learning_rate=params["lr"])
        else:
            opt = keras.optimizers.SGD(learning_rate=params["lr"])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        history = model.fit(
            train_ds,
            epochs=params["epochs"],
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=0,
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics({
                "epoch": epoch + 1,
                "train_loss": history.history["loss"][epoch],
                "train_acc": history.history["sparse_categorical_accuracy"][epoch],
                "val_loss": history.history["val_loss"][epoch],
                "val_acc": history.history["val_sparse_categorical_accuracy"][epoch],
            }, step=epoch + 1)

        final_metrics = {
            "final_train_acc": history.history["sparse_categorical_accuracy"][-1],
            "final_val_acc": history.history["val_sparse_categorical_accuracy"][-1],
            "best_val_acc": max(history.history["val_sparse_categorical_accuracy"]),
            "best_epoch": (
                history.history["val_sparse_categorical_accuracy"]
                .index(max(history.history["val_sparse_categorical_accuracy"])) + 1
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
            {"confusion_matrix": cm.tolist()},
            artifact_file="confusion_matrix.json"
        )

        for i in range(params["num_classes"]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            mlflow.log_metrics({
                f"precision_class_{i}": precision,
                f"recall_class_{i}": recall,
            })

        keras_path = os.path.join(model_dir, f"{run_id}.keras")
        model.save(keras_path)
        mlflow.keras.log_model(model, artifact_path="model")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=2,
                             subplot_titles=("Accuracy", "Loss"))
            fig.add_trace(
                go.Scatter(
                    y=history.history["sparse_categorical_accuracy"],
                    name="train_acc", mode="lines+markers"
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["val_sparse_categorical_accuracy"],
                    name="val_acc", mode="lines+markers"
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["loss"],
                    name="train_loss", mode="lines+markers"
                ), row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history["val_loss"],
                    name="val_loss", mode="lines+markers"
                ), row=1, col=2
            )
            fig.update_layout(
                title_text=f"Training Run: {run_name}",
                showlegend=True
            )
            fig_path = os.path.join(model_dir, f"{run_id}_training_curve.html")
            fig.write_html(fig_path)
            mlflow.log_artifact(fig_path)
        except ImportError:
            pass

        return model, final_metrics, run_id
```

### 4.3 Grid Search

- [ ] Создать `training/scripts/train_grid.py`:

```python
import itertools
from scripts.train import train_single
from scripts.load_data import load_dataset
from sklearn.model_selection import train_test_split

PARAM_GRID = {
    "epochs": [20, 30],
    "batch_size": [16, 32, 64],
    "lr": [0.01, 0.005, 0.001],
    "optimizer": ["Adam", "SGD"],
    "aug_factor": [4, 6, 8],
    "noise_std": [0.005, 0.01],
    "dropout": [0.3, 0.5],
}

def grid_search(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    best_val_acc = 0
    best_params = None
    best_run_id = None
    results_table = []

    total = 1
    for v in values:
        total *= len(v)
    print(f"Total combinations: {total}")

    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        params["num_points"] = 4096
        params["num_classes"] = 3

        print(f"[{i+1}/{total}] {params}")
        model, metrics, run_id = train_single(params, X_train, y_train,
                                           X_test, y_test)

        row = {**params, **metrics, "run_id": run_id}
        results_table.append(row)

        if metrics["best_val_acc"] > best_val_acc:
            best_val_acc = metrics["best_val_acc"]
            best_params = params
            best_run_id = run_id

        print(f"  -> best_val_acc={metrics['best_val_acc']:.4f}")

    return best_params, best_val_acc, best_run_id, results_table

if __name__ == "__main__":
    X, y, classes = load_dataset()
    best_params, best_val_acc, best_run_id, results = grid_search(X, y)
    print(f"\nBest: {best_params}")
    print(f"Best val_acc: {best_val_acc}")
    print(f"Best run_id: {best_run_id}")
```

**Чеклист MLflow:**

- [ ] Эксперимент создаётся в `training/data/mlruns/`
- [ ] `mlflow ui` открывает веб-интерфейс
- [ ] Каждый run логирует параметры
- [ ] Метрики видны по эпохам
- [ ] Confusion matrix сохранена как artifact
- [ ] Training curves (HTML) сохранены как artifact
- [ ] `.keras`-модель сохранена как artifact
- [ ] Best run определяется по `best_val_acc`

---

## Этап 5 — Экспорт в ONNX

- [ ] Установить: `pip install tf2onnx onnx onnxruntime`

- [ ] Создать `training/scripts/export_onnx.py`:

```python
import keras
import tf2onnx
import mlflow
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.mlflow_setup import setup_mlflow

def export_onnx(run_id: str, output_dir: str = "training/data/models"):
    os.makedirs(output_dir, exist_ok=True)
    setup_mlflow()

    model_uri = f"runs:/{run_id}/model"
    keras_model = mlflow.keras.load_model(model_uri)

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
```

**Чеклист ONNX:**

- [ ] `.onnx`-файл создаётся
- [ ] Размер ~2–3 MB
- [ ] ONNX валиден: `onnx.checker.check_model(model)`
- [ ] Инференс ONNX vs Keras даёт <1% diff

---

## Этап 6 — Тестирование

- [ ] Создать `tests/test_model.py`:

```python
import numpy as np
import sys
sys.path.insert(0, "training")
from models.pointnet import build_pointnet

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
```

- [ ] Создать `tests/test_onnx_inference.py`:

```python
import numpy as np
import onnx
import onnxruntime as ort

def test_onnx_inference(onnx_path: str):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 4096, 3).astype(np.float32)
    out = sess.run(None, {"input_layer": dummy})
    assert out[0].shape == (1, 3)
    assert np.isclose(out[0].sum(), 1.0, atol=0.01)

if __name__ == "__main__":
    import sys
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "training/data/models/best.onnx"
    test_onnx_inference(onnx_path)
    print("ONNX test passed!")
```

**Чеклист тестирования:**

- [ ] `python -m pytest tests/ -v` проходит
- [ ] ONNX инференс согласован с Keras

---

## Этап 7 — Docker MLflow (опционально)

- [ ] Создать `docker-compose.yml`:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    volumes:
      - ./training/data/mlruns:/mlflow/data
    command: mlflow server --backend-store-uri /mlflow/data --host 0.0.0.0
```

**Чеклист Docker:**

- [ ] `docker compose up -d` запускает MLflow
- [ ] UI доступен на `:5000`
- [ ] Данные сохраняются локально в `training/data/mlruns/`

---

## Этап 8 — Makefile

- [ ] Создать `Makefile`:

```makefile
.PHONY: train train-grid export-onnx test mlflow-ui

TRAINING_DIR = training
DATA_DIR = training/data
MODELS_DIR = training/data/models

train:
\tpython -m $(TRAINING_DIR).scripts.train_grid

export-onnx:
\tpython -m $(TRAINING_DIR).scripts.export_onnx --run-id=$(RUN_ID)

test:
\tpython -m pytest tests/ -v

mlflow-ui:
\tmlflow ui --backend-store-uri file://$(PWD)/$(DATA_DIR)/mlruns

clean:
\trm -rf $(DATA_DIR)/models/*.keras $(DATA_DIR)/models/*.onnx
\trm -rf $(DATA_DIR)/mlruns
```

**Чеклист Makefile:**

- [ ] `make train` — запуск перебора
- [ ] `make export-onnx RUN_ID=<id>` — экспорт модели
- [ ] `make test` — запуск тестов
- [ ] `make mlflow-ui` — открыть MLflow UI

---

## Итоговая таблица результатов

| epochs | batch | lr | optimizer | aug | noise | dropout | best_val_acc | overfit_gap | run_id |
|--------|-------|----|----------|-----|-------|---------|-------------|-------------|--------|
| базовый | 64 | 0.001 | SGD | 4 | 0.005 | 0.3 | 0.54 | 17 | — |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## Ожидаемые результаты

| Метрика | До | Цель |
|---------|----|------|
| Validation accuracy | ~54% | >75% |
| Overfit gap | ~17 | <5 |
| Per-class balanced accuracy | — | >70% каждый |
| ONNX model size | — | <5 MB |
| Inference time | — | <100ms |

---

## Контрольные вопросы (для защиты)

1. Архитектура PointNet
2. Что такое плотное облако точек? Форматы представления
3. Размер карты признаков и количество каналов в PointNet
4. В PointNet используются какие слои CNN или MLP?
5. Самостоятельно посчитать по Confusion Matrix: Accuracy, Precision, Recall