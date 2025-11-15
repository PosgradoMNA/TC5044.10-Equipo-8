import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import mlflow.sklearn
from .config import MLFLOW_TRACKING_URI
import mlflow

class DataDriftMonitor:
    def __init__(self, baseline_data, models):
        self.baseline_data = baseline_data
        self.models = models
        self.baseline_metrics = {}
        self.drift_thresholds = {
            'r2_drop': 0.1,
            'rmse_increase': 0.2,
            'ks_test_p': 0.05
        }
        
    def calculate_baseline_metrics(self, X_test, Y_test):
        """Calculate baseline performance metrics."""
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            self.baseline_metrics[name] = {
                'r2_heating': r2_score(Y_test['heating_load'], predictions[:, 0]),
                'r2_cooling': r2_score(Y_test['cooling_load'], predictions[:, 1]),
                'rmse_heating': np.sqrt(mean_squared_error(Y_test['heating_load'], predictions[:, 0])),
                'rmse_cooling': np.sqrt(mean_squared_error(Y_test['cooling_load'], predictions[:, 1]))
            }
        print("Baseline metrics calculated")
        
    def simulate_drift_scenarios(self, X_baseline, Y_baseline):
        """Generate different drift scenarios."""
        scenarios = {}
        
        X_shift = X_baseline.copy()
        X_shift['relative_compactness'] += 0.1
        X_shift['glazing_area'] += 0.05
        scenarios['efficiency_improvement'] = (X_shift, Y_baseline)
        
        X_missing = X_baseline.copy()
        X_missing['orientation'] = X_missing['orientation'].mean()
        scenarios['orientation_corruption'] = (X_missing, Y_baseline)
        
        X_seasonal = X_baseline.copy()
        X_seasonal['overall_height'] *= 1.2
        X_seasonal['wall_area'] *= 1.15
        scenarios['winter_buildings'] = (X_seasonal, Y_baseline)
        
        X_extreme = X_baseline.copy()
        X_extreme['relative_compactness'] *= 0.7
        X_extreme['surface_area'] *= 1.3
        X_extreme['glazing_area'] *= 0.5
        scenarios['new_building_types'] = (X_extreme, Y_baseline)
        
        return scenarios
        
    def detect_statistical_drift(self, X_baseline, X_new):
        """Detect drift using statistical tests."""
        drift_results = {}
        
        for column in X_baseline.columns:
            if column in X_new.columns:
                ks_stat, p_value = stats.ks_2samp(X_baseline[column], X_new[column])
                drift_results[column] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < self.drift_thresholds['ks_test_p']
                }
                
        return drift_results
        
    def evaluate_performance_drift(self, X_new, Y_new, scenario_name):
        """Evaluate model performance on drifted data."""
        results = {}
        
        for name, model in self.models.items():
            predictions = model.predict(X_new)
            
            current_metrics = {
                'r2_heating': r2_score(Y_new['heating_load'], predictions[:, 0]),
                'r2_cooling': r2_score(Y_new['cooling_load'], predictions[:, 1]),
                'rmse_heating': np.sqrt(mean_squared_error(Y_new['heating_load'], predictions[:, 0])),
                'rmse_cooling': np.sqrt(mean_squared_error(Y_new['cooling_load'], predictions[:, 1]))
            }
            
            baseline = self.baseline_metrics[name]
            degradation = {
                'r2_heating_drop': baseline['r2_heating'] - current_metrics['r2_heating'],
                'r2_cooling_drop': baseline['r2_cooling'] - current_metrics['r2_cooling'],
                'rmse_heating_increase': (current_metrics['rmse_heating'] - baseline['rmse_heating']) / baseline['rmse_heating'],
                'rmse_cooling_increase': (current_metrics['rmse_cooling'] - baseline['rmse_cooling']) / baseline['rmse_cooling']
            }
            
            alerts = {
                'r2_heating_alert': degradation['r2_heating_drop'] > self.drift_thresholds['r2_drop'],
                'r2_cooling_alert': degradation['r2_cooling_drop'] > self.drift_thresholds['r2_drop'],
                'rmse_heating_alert': degradation['rmse_heating_increase'] > self.drift_thresholds['rmse_increase'],
                'rmse_cooling_alert': degradation['rmse_cooling_increase'] > self.drift_thresholds['rmse_increase']
            }
            
            results[name] = {
                'current_metrics': current_metrics,
                'degradation': degradation,
                'alerts': alerts,
                'action_required': any(alerts.values())
            }
            
        return results
        
    def generate_drift_report(self, scenario_name, statistical_drift, performance_drift):
        """Generate drift report."""
        print(f"\n{'='*60}")
        print(f"DRIFT MONITORING REPORT: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        print("\nSTATISTICAL DRIFT DETECTION:")
        drift_detected = sum(1 for result in statistical_drift.values() if result['drift_detected'])
        print(f"Features with significant drift: {drift_detected}/{len(statistical_drift)}")
        
        for feature, result in statistical_drift.items():
            if result['drift_detected']:
                print(f"  WARNING {feature}: KS-stat={result['ks_statistic']:.3f}, p-value={result['p_value']:.4f}")
        
        print("\nPERFORMANCE IMPACT:")
        for model_name, results in performance_drift.items():
            print(f"\n{model_name}:")
            degradation = results['degradation']
            alerts = results['alerts']
            
            status_heating_r2 = "ALERT" if alerts['r2_heating_alert'] else "OK"
            status_cooling_r2 = "ALERT" if alerts['r2_cooling_alert'] else "OK"
            status_heating_rmse = "ALERT" if alerts['rmse_heating_alert'] else "OK"
            status_cooling_rmse = "ALERT" if alerts['rmse_cooling_alert'] else "OK"
            
            print(f"  R2 Heating Drop: {degradation['r2_heating_drop']:.3f} [{status_heating_r2}]")
            print(f"  R2 Cooling Drop: {degradation['r2_cooling_drop']:.3f} [{status_cooling_r2}]")
            print(f"  RMSE Heating Increase: {degradation['rmse_heating_increase']:.1%} [{status_heating_rmse}]")
            print(f"  RMSE Cooling Increase: {degradation['rmse_cooling_increase']:.1%} [{status_cooling_rmse}]")
            
            if results['action_required']:
                print(f"  STATUS: ACTION REQUIRED - Performance degradation detected")
            else:
                print(f"  STATUS: Performance within acceptable limits")
        
        print(f"\nRECOMMENDED ACTIONS:")
        action_required = any(results['action_required'] for results in performance_drift.values())
        
        if action_required:
            print("  1. Retrain models with recent data")
            print("  2. Review feature engineering pipeline")
            print("  3. Collect more data from new distribution")
            print("  4. Update data preprocessing steps")
        else:
            print("  No immediate action required")
            print("  Continue monitoring for trends")
            
    def plot_drift_comparison(self, X_baseline, scenarios):
        """Plot feature distributions for drift visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        key_features = ['relative_compactness', 'glazing_area', 'overall_height', 'surface_area']
        
        for i, feature in enumerate(key_features):
            ax = axes[i]
            
            ax.hist(X_baseline[feature], alpha=0.5, label='Baseline', bins=20, color='blue')
            
            colors = ['red', 'green', 'orange', 'purple']
            for j, (scenario_name, (X_drift, _)) in enumerate(scenarios.items()):
                ax.hist(X_drift[feature], alpha=0.3, label=scenario_name, bins=20, color=colors[j])
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('reports/figures/drift_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_drift_monitoring(self, X_test, Y_test):
        """Run complete drift monitoring pipeline."""
        print("Starting Data Drift Monitoring...")
        
        self.calculate_baseline_metrics(X_test, Y_test)
        
        scenarios = self.simulate_drift_scenarios(X_test, Y_test)
        
        import os
        os.makedirs('reports/figures', exist_ok=True)
        
        all_results = {}
        for scenario_name, (X_drift, Y_drift) in scenarios.items():
            statistical_drift = self.detect_statistical_drift(X_test, X_drift)
            
            performance_drift = self.evaluate_performance_drift(X_drift, Y_drift, scenario_name)
            
            self.generate_drift_report(scenario_name, statistical_drift, performance_drift)
            
            all_results[scenario_name] = {
                'statistical_drift': statistical_drift,
                'performance_drift': performance_drift
            }
        
        self.plot_drift_comparison(X_test, scenarios)
        
        return all_results
