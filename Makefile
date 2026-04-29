.PHONY: help train train-default export-onnx test mlflow-ui clean

TRAINING_DIR = training
DATA_DIR = training/data
MODELS_DIR = training/data/models

help:
	@echo "Доступные команды:"
	@echo "  make train-default             - обучить модель с параметрами по умолчанию (epochs=20, bs=32, lr=0.001)"
	@echo "  make train EPOCHS=30 BATCH_SIZE=64 LR=0.005 - обучить с своими гиперпараметрами"
	@echo "  make mlflow-ui                 - открыть MLflow UI (http://localhost:5000)"
	@echo "  make test                      - запустить тесты"
	@echo "  make clean                     - очистить модели и логи"
	@echo ""
	@echo "Примеры:"
	@echo "  make train EPOCHS=30 BATCH_SIZE=16 LR=0.001"
	@echo "  make train EPOCHS=30 BATCH_SIZE=64 LR=0.005 OPTIMIZER=SGD DROPOUT=0.5"
	@echo "  make train EPOCHS=50 BATCH_SIZE=64  # автоопределение GPU"

train-default:
	python3 -m training.scripts.train_cli --epochs=20 --batch-size=32 --lr=0.001 --optimizer=Adam --dropout=0.3

train:
	python3 -m training.scripts.train_cli --epochs=$(or $(EPOCHS),20) --batch-size=$(or $(BATCH_SIZE),32) --lr=$(or $(LR),0.001) --optimizer=$(or $(OPTIMIZER),Adam) --dropout=$(or $(DROPOUT),0.3)

export-onnx:
	python3 -m $(TRAINING_DIR).scripts.export_onnx --run-id=$(RUN_ID)

test:
	python3 -m pytest tests/ -v

mlflow-ui:
	mlflow ui --backend-store-uri file://$(PWD)/$(DATA_DIR)/mlruns

clean:
	rm -rf $(DATA_DIR)/models/*.keras $(DATA_DIR)/models/*.onnx
	rm -rf $(DATA_DIR)/mlruns