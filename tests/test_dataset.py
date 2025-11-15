import subprocess
import pandas as pd
from energy_efficiency.dataset import DataLoader


class TestDataLoader:
    def test_get_dataframe_from_file(self, sample_csv_file):
        """Test CSV loading and column renaming."""
        loader = DataLoader()
        df = loader.getDataFrameFromFile(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "relative_compactness" in df.columns
        assert "heating_load" in df.columns
        assert "X1" not in df.columns

    def test_save_dataframe_as_file(self, sample_data, temp_dir):
        """Test DataFrame saving functionality."""
        loader = DataLoader()
        test_file = "test_output.csv"

        original_run = subprocess.run
        subprocess.run = lambda *args, **kwargs: None

        try:
            loader.saveDataFrameAsFileWithDVC(sample_data, temp_dir, test_file)
            saved_file = temp_dir / test_file
            assert saved_file.exists()

            loaded_df = pd.read_csv(saved_file)
            assert len(loaded_df) == len(sample_data)
        finally:
            subprocess.run = original_run
