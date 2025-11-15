import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def sample_data():
    """Create sample energy efficiency data for testing."""
    np.random.seed(42)
    data = {
        'X1': np.random.uniform(0.6, 1.0, 100),
        'X2': np.random.uniform(500, 800, 100),
        'X3': np.random.uniform(200, 400, 100),
        'X4': np.random.uniform(100, 250, 100),
        'X5': np.random.uniform(3, 7, 100),
        'X6': np.random.choice([2, 3, 4, 5], 100),
        'X7': np.random.uniform(0, 0.4, 100),
        'X8': np.random.choice([0, 1, 2, 3, 4, 5], 100),
        'Y1': np.random.uniform(10, 40, 100),
        'Y2': np.random.uniform(10, 45, 100)
    }
    data['X1'][5] = np.nan
    data['Y1'][10] = 100
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_csv_file(sample_data, temp_dir):
    """Create sample CSV file for testing."""
    csv_path = temp_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path
