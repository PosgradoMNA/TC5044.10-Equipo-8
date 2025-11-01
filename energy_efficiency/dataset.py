import pandas as pd
from pathlib import Path
import subprocess
from .config import RAW_DATA_DIR, RAW_DATA_FILE, NAMING_MAP


class DataLoader:
    def getBaseDir(self):
        """
        Get the base directory path for the current file.

        Returns the parent directory of the current file as a Path object.
        """
        return Path(__file__).resolve().parent

    def getDataFrameFromFile(self, route_to_file: str):
        """
        Load a CSV file and return a pandas DataFrame with renamed columns.

        Keyword arguments:
        route_to_file -- relative path to the CSV file from the base directory

        Returns a DataFrame with columns renamed according to NAMING_MAP.
        """
        file_path = Path(route_to_file)
        df = pd.read_csv(file_path)
        print(f"\nSuccesfully loaded DF from {file_path}...", "\n")
        return df.rename(columns=NAMING_MAP, inplace=False)

    def saveDataFrameAsFileWithDVC(self, df: pd.DataFrame, route: str, file_name: str):
        """
        Save a DataFrame to CSV file and add it to DVC version control.

        Keyword arguments:
        df -- pandas DataFrame to save
        route -- directory path where to save the file
        file_name -- name of the CSV file to create

        Creates the directory if it doesn't exist and adds the file to DVC tracking.
        """
        folder_path = Path(route)
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / file_name
        df.to_csv(file_path, index=False)
        print(f"\nSuccesfully saved DF in {file_path}...", "\n")

        print("\nInitializing DVC versioning...", "\n")
        subprocess.run(["dvc", "add", str(file_path)], check=True)


if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.getDataFrameFromFile(RAW_DATA_DIR / RAW_DATA_FILE)
    print(f"Dataset loaded with shape: {df.shape}")
