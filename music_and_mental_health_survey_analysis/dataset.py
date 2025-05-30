import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from kaggle.api.kaggle_api_extended import KaggleApi

from music_and_mental_health_survey_analysis.config \
    import PROCESSED_DATA_DIR, RAW_DATA_DIR, KAGGLE_DATASET

app = typer.Typer()

def download_data():
    """
    Downloads the 'mxmh_survey_results.csv' dataset from Kaggle
    if it does not already exist in the RAW_DATA_DIR. Sets up Kaggle API 
    credentials, authenticates, and downloads the dataset, 
    unzipping it into the specified directory.

    Returns:
        None
    Logs:
        - Info message if the dataset already exists and download is skipped.
    """
    if (RAW_DATA_DIR / 'mxmh_survey_results.csv').exists():
        logger.info("Dataset already exists. Skipping download.")
        return

    # Setup credentials
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

    # Download dataset
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=KAGGLE_DATASET,
        path=RAW_DATA_DIR,
        unzip=True
    )

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    logger.info('Downloading dataset...')
    download_data()
    logger.success('Done.')

if __name__ == "__main__":
    app()
