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

#################################################################################
# PROJECT RULES                                                                #
#################################################################################

## Make complete pipeline (with MLflow UI)
pipeline:
	$(PYTHON_INTERPRETER) -m energy_efficiency.main
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
