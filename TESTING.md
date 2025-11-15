# Testing Guide

## Overview
This project includes unit and integration tests to validate critical components and end-to-end pipeline functionality.

## Test Structure
```
tests/
├── __init__.py
├── conftest.py           # Test fixtures and configuration
├── test_dataset.py       # Unit tests for data loading
├── test_features.py      # Unit tests for preprocessing
├── test_modeling.py      # Unit tests for training/evaluation
└── test_integration.py   # Integration tests for full pipeline
```

## Running Tests

### All Tests (Recommended)
```bash
make test
# or
pytest -q
```

### Unit Tests Only
```bash
make test_unit
# or
pytest tests/test_dataset.py tests/test_features.py tests/test_modeling.py -q
```

### Integration Tests Only
```bash
make test_integration
# or
pytest tests/test_integration.py -q
```

### With Coverage Report
```bash
make test_coverage
# or
pytest --cov=energy_efficiency --cov-report=html
```

## Test Coverage

### Unit Tests
- **DataLoader**: CSV loading, column renaming, file saving
- **DataPreprocessor**: Numeric conversion, imputation, outlier detection
- **ModelTrainer**: Initialization, data splitting, preprocessor creation
- **ModelEvaluator**: Metric calculation, model evaluation

### Integration Tests
- **End-to-End Pipeline**: Complete flow from data loading to evaluation
- **Data Consistency**: Validates data integrity through pipeline stages
- **Minimal Dataset Handling**: Tests pipeline robustness with small datasets

## Test Fixtures
- `sample_data`: Synthetic energy efficiency dataset with missing values and outliers
- `temp_dir`: Temporary directory for file operations
- `sample_csv_file`: CSV file for testing data loading

## Dependencies
Tests require pytest which is included in requirements.txt:
```bash
pip install pytest
```

## Continuous Integration
Tests are designed to run in CI/CD environments with mocked external dependencies (MLflow, DVC).
