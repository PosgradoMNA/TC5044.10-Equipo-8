# Data Drift Monitoring System

## Overview
Data drift detection system that simulates distribution changes and monitors model performance degradation with automated alerting.

## Features

### Drift Simulation Scenarios
1. **Efficiency Improvement**: Buildings become more compact with increased glazing
2. **Orientation Corruption**: Missing orientation data (all buildings same orientation)  
3. **Winter Buildings**: Seasonal changes with taller buildings and more wall area
4. **New Building Types**: Extreme drift with different building characteristics

### Statistical Drift Detection
- **Kolmogorov-Smirnov Test**: Detects distribution changes between baseline and new data
- **P-value Threshold**: 0.05 for statistical significance
- **Feature-level Analysis**: Individual drift detection per feature

### Performance Impact Monitoring
- **R² Drop Threshold**: Alert if R² decreases by >10%
- **RMSE Increase Threshold**: Alert if RMSE increases by >20%
- **Multi-model Evaluation**: Tests all trained models (LinearRegression, RandomForest, GradientBoosting)

### Alert System
- **Statistical Alerts**: Features with significant distribution drift
- **Performance Alerts**: Models exceeding degradation thresholds
- **Action Required**: Automated recommendations when thresholds exceeded

## Usage

### Run Drift Monitoring
```bash
make drift_monitor
```

### Programmatic Usage
```python
from energy_efficiency.drift_monitor import DataDriftMonitor

monitor = DataDriftMonitor(baseline_data, models)
results = monitor.run_drift_monitoring(X_test, Y_test)
```

## Output

### Console Reports
- Statistical drift detection results
- Performance impact analysis per model
- Alert status for each metric
- Automated action recommendations

### Visualizations
- Feature distribution comparisons
- Saved to `reports/figures/drift_comparison.png`
- Baseline vs drift scenario histograms

## Alert Thresholds

| Metric | Threshold | Action |
|--------|-----------|---------|
| R² Drop | >10% | Retrain models |
| RMSE Increase | >20% | Review pipeline |
| KS Test p-value | <0.05 | Statistical drift detected |

## Recommended Actions

When drift is detected:
1. **Retrain models** with recent data
2. **Review feature pipeline** for data quality issues
3. **Collect more data** from new distribution
4. **Update preprocessing** steps if needed

## Example Output

```
DRIFT MONITORING REPORT: EFFICIENCY_IMPROVEMENT
Statistical drift detected in 2/9 features
- relative_compactness: KS-stat=0.451, p-value=0.0000
- glazing_area: KS-stat=0.366, p-value=0.0000

Performance Impact:
RandomForest: R² Drop=0.733 [ALERT], RMSE Increase=326% [ALERT]
STATUS: ACTION REQUIRED - Performance degradation detected
```

## Integration
- **Automated Testing**: Included in test suite
- **Make Command**: `make drift_monitor`
- **CI/CD Ready**: Can be integrated into deployment pipelines
- **Threshold Configuration**: Easily adjustable alert thresholds
