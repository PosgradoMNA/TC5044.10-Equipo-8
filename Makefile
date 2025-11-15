.PHONY: data train clean requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = energy_efficiency
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) energy_efficiency/dataset.py

## Train models
train:
	$(PYTHON_INTERPRETER) -m energy_efficiency.modeling.train

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Download data from S3
sync_data_from_s3:
	$(PYTHON_INTERPRETER) -m dvc pull

## Upload data to S3
sync_data_to_s3:
	$(PYTHON_INTERPRETER) -m dvc push

#################################################################################
# PROJECT RULES                                                                #
#################################################################################

## Make complete pipeline (pull data, process, train, push results)
pipeline: sync_data_from_s3
	$(PYTHON_INTERPRETER) -m energy_efficiency.main
	$(PYTHON_INTERPRETER) -m dvc push
	@echo "Running unit tests..."
	$(PYTHON_INTERPRETER) -m pytest tests/test_dataset.py tests/test_features.py tests/test_modeling.py tests/test_api.py tests/test_drift_monitor.py -q
	@echo "Running integration tests..."
	$(PYTHON_INTERPRETER) -m pytest tests/test_integration.py -q
	@echo "Pipeline complete! Starting MLflow UI..."
	@echo "Open http://127.0.0.1:5000 in your browser"
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

## Quick pipeline without S3 sync
pipeline_local:
	$(PYTHON_INTERPRETER) -m energy_efficiency.main
	@echo "Running unit tests..."
	$(PYTHON_INTERPRETER) -m pytest tests/test_dataset.py tests/test_features.py tests/test_modeling.py tests/test_api.py tests/test_drift_monitor.py -q
	@echo "Running integration tests..."
	$(PYTHON_INTERPRETER) -m pytest tests/test_integration.py -q
	@echo "Pipeline complete! Starting MLflow UI..."
	@echo "Open http://127.0.0.1:5000 in your browser"
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
## Start MLflow UI
mlflow_ui:
	mlflow ui --backend-store-uri ./mlruns

## Run all tests
test:
	$(PYTHON_INTERPRETER) -m pytest -q

## Run unit tests only
test_unit:
	$(PYTHON_INTERPRETER) -m pytest tests/test_dataset.py tests/test_features.py tests/test_modeling.py tests/test_api.py tests/test_drift_monitor.py -q

## Run integration tests only
test_integration:
	$(PYTHON_INTERPRETER) -m pytest tests/test_integration.py -q

## Run tests with coverage
test_coverage:
	$(PYTHON_INTERPRETER) -m pytest --cov=energy_efficiency --cov-report=html

## Start FastAPI server
serve:
	$(PYTHON_INTERPRETER) -m energy_efficiency.serve

## Start FastAPI server with custom host/port
serve_custom:
	$(PYTHON_INTERPRETER) -c "from energy_efficiency.serve import start_server; start_server('0.0.0.0', 8000)"

## Build Docker image
docker_build:
	docker build -t ml-service:latest .

## Run Docker container
docker_run:
	docker run -p 8000:8000 ml-service:latest

## Build and run with docker-compose
docker_compose:
	docker-compose up --build

## Run drift monitoring
drift_monitor:
	$(PYTHON_INTERPRETER) -m energy_efficiency.run_drift_monitoring
