.PHONY: train train-grid export-onnx test mlflow-ui

TRAINING_DIR = training
DATA_DIR = training/data
MODELS_DIR = training/data/models

train:
	python3 -m $(TRAINING_DIR).scripts.train_grid

export-onnx:
	python3 -m $(TRAINING_DIR).scripts.export_onnx --run-id=$(RUN_ID)

test:
	python3 -m pytest tests/ -v

mlflow-ui:
	mlflow ui --backend-store-uri file://$(PWD)/$(DATA_DIR)/mlruns

clean:
	rm -rf $(DATA_DIR)/models/*.keras $(DATA_DIR)/models/*.onnx
	rm -rf $(DATA_DIR)/mlruns