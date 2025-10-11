from handlers.model_evaluator import ModelEvaluator
from handlers.model_trainer import ModelTrainer
from handlers.visual_eda import VisualEDA
from handlers.data_loader import DataLoader
from handlers.data_preprocessor import DataPreprocessor


def main():
    data_loader = DataLoader()

    df = data_loader.getDataFrameFromFile("data/energy_efficiency_modified.csv")

    print(f"\n\n > Lodaded data - Rows: {df.shape[0]}, Columns: {df.shape[1]}", "\n\n")

    eda = VisualEDA(df)
    data_preprocessor = DataPreprocessor(df)

    data_preprocessor.convert_numeric()

    print(f"\n\n > Data overview before cleansing...", "\n\n")

    eda.overview()

    print(f"\n\n > Initializing visual EDA...", "\n\n")

    eda.plot_histograms()
    eda.plot_boxplots()
    eda.plot_correlation_heatmap()

    print(f"\n\n > Initializing data cleansing...", "\n\n")

    data_preprocessor.impute_missing()
    data_preprocessor.detect_outliers()
    data_preprocessor.standardize()

    print(f"\n\n > Data overview after cleansing...", "\n\n")

    eda.overview()

    data_loader.saveDataFrameAsFileWithDVC(
        data_preprocessor.df, "data/cleansed", "energy_efficiency_modified.csv"
    )

    print(f"\n\n > Initializing model training...", "\n\n")

    trainer = ModelTrainer(data_preprocessor.df)
    trainer.split_and_scale()
    trainer.train_models()

    print(f"\n\n > Initializing model evaluation...", "\n\n")

    evaluator = ModelEvaluator(trainer.models, trainer.X_test_scaled, trainer.Y_test)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
