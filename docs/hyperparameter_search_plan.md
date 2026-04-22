# Декомпозиционный план: перебор гиперпараметров PointNet

## 1. Описание задачи

На основе ноутбука `tree_classification.ipynb` составить скрипт
для систематического перебора гиперпараметров модели PointNet с целью
улучшения метрик классификации пород деревьев по облакам точек.

**Текущие результаты (базовая модель):**

- Train accuracy: ~71%
- Validation accuracy: ~54%
- Явное переобучение: gap ~17%
- Confusion matrix: слабая дискриминация классов

---

## 2. Подготовка данных

### 2.1 Константы

```python
DATA_DIR = "./content"
H5_FILE = "v2.h5"
NUM_POINTS = 4096
NUM_CLASSES = 3
```

### 2.2 Классы

- 0: Береза
- 1: Ель
- 2: Сосна

### 2.3 Разбиение выборки

```python
skf = StratifiedKFold(n_splits=5)
X_train, X_test — train/test split
X_augment, y_augment — аугментированные данные
```

### 2.4 Аугментация данных

Для каждого семпла из train:

1. Random sampling: `np.random.choice(NUM_POINTS, size=NUM_POINTS, replace=True)`
2. Гауссов шум: `+ np.random.normal(0, noise_std, shape)`

Количество аугментаций варьируется.

---

## 3. Архитектура PointNet

### 3.1 Базовые блоки

```python
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
```

### 3.2 T-Net (трансформер)

```python
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
```

### 3.3 Модель классификации

```
inputs: Input(shape=(NUM_POINTS, 3))

T-Net(3) → input transform
Conv1D(32) → BN → ReLU
Conv1D(32) → BN → ReLU
T-Net(32) → feature transform
Conv1D(32) → BN → ReLU
Conv1D(64) → BN → ReLU
Conv1D(512) → BN → ReLU
GlobalMaxPooling1D
Dense(256) → BN → ReLU → Dropout(rate)
Dense(128) → BN → ReLU → Dropout(rate)
Dense(NUM_CLASSES, activation="softmax")
```

Общее количество параметров: 748,076

---

## 4. Перебор гиперпараметров

### 4.1 Параметры для перебора

| Параметр | Базовое | Варианты | Описание |
|----------|---------|----------|-----------|
| `EPOCHS` | 10 | 10, 20, 30, 50 | Количество эпох обучения |
| `BATCH_SIZE` | 64 | 16, 32, 64 | Размер батча |
| `LEARNING_RATE` | 0.001 | 0.01, 0.005, 0.001, 0.0005 | Скорость обучения |
| `OPTIMIZER` | SGD | SGD, Adam | Оптимизатор |
| `AUG_FACTOR` | 4 | 2, 4, 6, 8 | Множитель аугментации |
| `NOISE_STD` | 0.005 | 0.001, 0.005, 0.01 | Стандартное отклонение шума |
| `DROPOUT` | 0.3 | 0.2, 0.3, 0.5 | Rate dropout слоев |

### 4.2 Стратегия перебора

#### Вариант A: Полный перебор (Grid Search)

Общее количество комбинаций:

```
4 (epochs) × 3 (batch) × 4 (lr) × 2 (optimizer) × 4 (aug) × 3 (noise) × 3 (dropout)
= 3456 комбинаций
```

Примерное время: очень долго (~недели на CPU).

#### Вариант B: Частичный перебор (Batch Search)

Перебор по этапам:

1. **Этап 1** — optimizer + learning rate (6 комбинаций)
2. **Этап 2** — batch size + epochs (12 комбинаций)
3. **Этап 3** — augmentation + noise (12 комбинаций)
4. **Этап 4** — dropout (3 комбинации)

Итого: ~33 комбинации на каждом этапе, лучшие переносятся дальше.

#### Вариант C: Случайный перебор (Random Search)

```python
n_trials = 50  # или 100
for trial in range(n_trials):
    params = {
        'epochs': random.choice([10, 20, 30, 50]),
        'batch_size': random.choice([16, 32, 64]),
        'lr': random.uniform(0.0001, 0.02),
        'optimizer': random.choice(['SGD', 'Adam']),
        'aug_factor': random.choice([2, 4, 6, 8]),
        'noise_std': random.uniform(0.001, 0.015),
        'dropout': random.choice([0.2, 0.3, 0.5]),
    }
    train_and_evaluate(params)
```

Рекомендуется: **Вариант B** — последовательный перебор
с наследованием лучших параметров.

---

## 5. Цикл обучения

```python
def train_and_evaluate(params):
    # 1. Аугментация
    X_augment = augment_data(X_train, params['aug_factor'], params['noise_std'])

    # 2. Формирование датасета
    train_dataset = tf.data.Dataset.from_tensor_slices((X_augment, y_augment))
    train_dataset = train_dataset.shuffle(len(X_augment)).batch(params['batch_size'])
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(params['batch_size'])

    # 3. Сборка модели
    model = build_pointnet(params['dropout'])

    # 4. Компиляция
    if params['optimizer'] == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=params['lr'])
    else:
        opt = keras.optimizers.SGD(learning_rate=params['lr'])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["sparse_categorical_accuracy"],
    )

    # 5. Обучение
    history = model.fit(
        train_dataset,
        epochs=params['epochs'],
        validation_data=test_dataset,
        verbose=0,
    )

    # 6. Оценка
    results = {
        'train_acc': history.history['sparse_categorical_accuracy'][-1],
        'val_acc': history.history['val_sparse_categorical_accuracy'][-1],
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'overfit_gap': history.history['loss'][-1] - history.history['val_loss'][-1],
        'best_val_acc': max(history.history['val_sparse_categorical_accuracy']),
    }

    return model, results
```

---

## 6. Метрики для сравнения

### 6.1 Основные метрики

- `val_sparse_categorical_accuracy` — Accuracy на валидации
- `best_val_acc` — лучшее значение за все эпохи
- `overfit_gap` = `train_loss` - `val_loss` — оценка переобучения
- `per_class_accuracy` — accuracy для каждого класса

### 6.2 Confusion Matrix

```python
preds = model.predict(test_dataset)
preds = tf.math.argmax(preds, -1)
cm = confusion_matrix(y_true=y_test, y_pred=preds)
```

Для каждой комбинации сохраняется:

```python
results = {
    'params': params,
    'val_acc': ...,
    'best_val_acc': ...,
    'overfit_gap': ...,
    'cm': cm,
    'time_per_epoch': ...,
}
```

---

## 7. Визуализация результатов

### 7.1 Итоговая таблица

| epochs | batch | lr | optimizer | aug | noise | dropout | val_acc | gap | time |
|--------|-------|----|----------|-----|-------|---------|---------|-----|------|
| 10 | 64 | 0.001 | SGD | 4 | 0.005 | 0.3 | 0.54 | 17 | 164s |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 7.2 Графики

1. **Learning curves** — accuracy по эпохам для топ-5 конфигов
2. **Loss curves** — loss по эпохам
3. **Confusion matrix** — для лучшей модели

### 7.3 Выводы

- Лучшая комбинация гиперпараметров
- Как каждый параметр влияет на метрики
- Рекомендации по улучшению модели

---

## 8. Ожидаемые результаты

| Метрика | До | Цель |
|---------|----|------|
| Validation accuracy | ~54% | >75% |
| Overfit gap | ~17 | <5 |
| Per-class accuracy ( balanced) | нет | >70% для каждого |

---

## 9. Контрольные вопросы (для защиты)

1. Архитектура PointNet
2. Что такое плотное облако точек? Форматы представления
3. Размер карты признаков и количество каналов в PointNet
4. В PointNet используются какие слои CNN или MLP?
5. Самостоятельно посчитать по Confusion Matrix: Accuracy, Precision, Recall