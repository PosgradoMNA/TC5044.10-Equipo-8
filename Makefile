.PHONY: data train clean requirements sync_data_from_s3 sync_data_to_s3

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
	$(PYTHON_INTERPRETER) -m pip install "dvc[s3]"

## Download data from S3
sync_data_from_s3:
	$(PYTHON_INTERPRETER) -m dvc pull

## Upload data to S3
sync_data_to_s3:
	$(PYTHON_INTERPRETER) -m dvc push

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

## Make complete pipeline (pull data, process, train, push results)
pipeline: sync_data_from_s3
	$(PYTHON_INTERPRETER) -m energy_efficiency.main
	$(PYTHON_INTERPRETER) -m dvc push

## Quick pipeline without S3 sync
pipeline_local:
	$(PYTHON_INTERPRETER) -m energy_efficiency.main

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
