from energy_efficiency.modeling.train import ModelTrainer
from energy_efficiency.drift_monitor import DataDriftMonitor
from energy_efficiency.dataset import DataLoader
from energy_efficiency.features import DataPreprocessor
from energy_efficiency.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILE
import mlflow

def main():
    """Run drift monitoring on trained models."""
    
    data_loader = DataLoader()
    df = data_loader.getDataFrameFromFile(PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
    
    preprocessor = DataPreprocessor(df)
    preprocessor.convert_numeric()
    preprocessor.impute_missing()
    preprocessor.detect_outliers()
    
    trainer = ModelTrainer(preprocessor.df)
    trainer.split_data()
    trainer.train_models()
    
    monitor = DataDriftMonitor(trainer.X_test, trainer.models)
    
    results = monitor.run_drift_monitoring(trainer.X_test, trainer.Y_test)
    
    print("\nDrift monitoring completed. Check reports/figures/drift_comparison.png for visualizations.")
    
    return results

if __name__ == "__main__":
    main()
