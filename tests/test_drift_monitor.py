import pytest
import pandas as pd
import numpy as np
from energy_efficiency.drift_monitor import DataDriftMonitor

class TestDataDriftMonitor:
    def test_drift_monitor_initialization(self):
        """Test drift monitor initialization."""
        baseline_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        models = {'test_model': None}
        
        monitor = DataDriftMonitor(baseline_data, models)
        
        assert monitor.drift_thresholds['r2_drop'] == 0.1
        assert monitor.drift_thresholds['rmse_increase'] == 0.2
        assert monitor.drift_thresholds['ks_test_p'] == 0.05

    def test_simulate_drift_scenarios(self):
        """Test drift scenario generation."""
        baseline_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        models = {'test_model': None}
        monitor = DataDriftMonitor(baseline_data, models)
        
        X_baseline = pd.DataFrame({
            'relative_compactness': [0.8, 0.9, 0.7],
            'glazing_area': [0.1, 0.2, 0.3],
            'overall_height': [5, 6, 7],
            'wall_area': [200, 250, 300],
            'surface_area': [600, 700, 800],
            'orientation': [2, 3, 4]
        })
        Y_baseline = pd.DataFrame({'heating_load': [20, 25, 30], 'cooling_load': [15, 20, 25]})
        
        scenarios = monitor.simulate_drift_scenarios(X_baseline, Y_baseline)
        
        assert len(scenarios) == 4
        assert 'efficiency_improvement' in scenarios
        assert 'orientation_corruption' in scenarios
        assert 'winter_buildings' in scenarios
        assert 'new_building_types' in scenarios

    def test_detect_statistical_drift(self):
        """Test statistical drift detection."""
        baseline_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        models = {'test_model': None}
        monitor = DataDriftMonitor(baseline_data, models)
        
        X_baseline = pd.DataFrame({'feature1': np.random.normal(0, 1, 100)})
        X_drifted = pd.DataFrame({'feature1': np.random.normal(2, 1, 100)})
        
        drift_results = monitor.detect_statistical_drift(X_baseline, X_drifted)
        
        assert 'feature1' in drift_results
        assert 'ks_statistic' in drift_results['feature1']
        assert 'p_value' in drift_results['feature1']
        assert 'drift_detected' in drift_results['feature1']
