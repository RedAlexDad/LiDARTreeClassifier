.PHONY: train train-grid export-onnx test mlflow-ui clean

TRAINING_DIR = training
DATA_DIR = training/data
MODELS_DIR = training/data/models

train:
	python3 train_with_params.py --epochs=$(EPOCHS) --batch-size=$(BATCH_SIZE) --lr=$(LR) --optimizer=$(OPTIMIZER) --dropout=$(DROPOUT)

train-default:
	python3 train_with_params.py --epochs=20 --batch-size=32 --lr=0.001 --optimizer=Adam --dropout=0.3

export-onnx:
	python3 -m $(TRAINING_DIR).scripts.export_onnx --run-id=$(RUN_ID)

test:
	python3 -m pytest tests/ -v

mlflow-ui:
	mlflow ui --backend-store-uri file://$(PWD)/$(DATA_DIR)/mlruns

clean:
	rm -rf $(DATA_DIR)/models/*.keras $(DATA_DIR)/models/*.onnx
	rm -rf $(DATA_DIR)/mlruns