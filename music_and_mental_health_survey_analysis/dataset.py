import os
from pathlib import Path

from loguru import logger
import typer

import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi

from music_and_mental_health_survey_analysis.config import (
    KAGGLE_DATASET,
    RAW_DATA_DIR
)

app = typer.Typer()

def encode_yes_no_cols(df: pd.DataFrame):
    """
    Encode columns with Yes/No values to 1s and 0s
    
    Params:
        df (DataFrame): Input DataFrame
    
    Returns:
        df (DataFrame): Processed DataFrame
    """
    logger.info("Encoding Yes/No columns to binary.")
    encoded_cols = []
    for column in df.columns:
        if set(df[column].dropna().unique()).issubset({'Yes', 'No'}):
            df[column] = df[column].map({'Yes': 1.0, 'No': 0.0})
            encoded_cols.append(column)

    logger.success(f"Successfully encoded columns: {', '.join(encoded_cols)}")
    return df

@app.command()
def main(
    output_path: Path = typer.Option(
        RAW_DATA_DIR,
        help='Where to save the processed dataset')
):
    """
    Downloads the project dataset from Kaggle.

    Uses KaggleAPI to download the dataset and save to RAW_DATA_DIR.

    Params:
        output_path (Path): Location to save the CSV file.
    """
    logger.info('Downloading dataset...')

    if (output_path / 'mxmh_survey_results.csv').exists():
        logger.info("Dataset already exists. Skipping download.")
        return

    # Setup credentials
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

    # Download dataset
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=KAGGLE_DATASET,
        path=output_path,
        unzip=True
    )

    logger.success('Done.')

if __name__ == "__main__":
    app()
