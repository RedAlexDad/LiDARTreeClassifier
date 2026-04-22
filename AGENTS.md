# AGENTS.md

## Команды

```bash
make train              # полный grid search (training/scripts/train_grid.py)
python train_quick.py  # быстрый один прогон для проверки
python -m pytest tests/ -v  # тесты
make mlflow-ui        # открыть MLflow UI (http://localhost:5000)
make export-onnx RUN_ID=<run-id>  # экспорт модели в ONNX
make clean           # удалить модели и mlruns
```

## Структура

```
./
├── content/           # исходные .h5 файлы (v1.h5, v2.h5 — точка данных)
├── training/
│   ├── data/        # рабочие данные, модели (.keras/.onnx), mlruns
│   │   ├── v2.h5
│   │   ├── classes.json
│   │   ├── models/
│   │   └── mlruns/
│   ├── models/      # архитектуры (pointnet.py)
│   └── scripts/     # train.py, train_grid.py, export_onnx.py, load_data.py, augmentation.py
├── tests/           # test_model.py, test_onnx_inference.py
└── tree_classification.ipynb  # ноутбук-источник (лабораторная)
```

- Классы: `0=Береза, 1=Ель, 2=Сосна`
- PointNet: input (4096, 3), Conv1D→BN→ReLU, T-Net трансформеры, GlobalMaxPooling, Dense→softmax
- ~748K параметров

## Подводные камни

- `train_quick.py` и `training/scripts/train_grid.py` используют `sys.path.insert(0, ...)` костыли для импорта — не полагайтесь на стандартный package import
  - `train_quick.py`: `sys.path.insert(0, "training")` — относительный импорт от корня проекта
  - `training/scripts/train.py`: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` — поднимается на уровень вверх
- `load_data.py` читает из `training/data/v2.h5` (копия из `content/`), а не из `content/`
- Grid search: 864 комбинации по умолчанию (2×3×3×2×3×2×2) — используйте `train_quick.py` для быстрой проверки
- `.keras` модели сохраняются с run_id из MLflow