import os
import json
from pathlib import Path

from loguru import logger
import typer

import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi

from music_and_mental_health_survey_analysis.config import (
    RAW_DATA_DIR,
    KAGGLE_DATASET,
    CONFIG_FILE,
    DOMAIN_RULES,
    PROCESSED_DATA_DIR
)

app = typer.Typer()

@app.command()
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
    logger.info('Downloading dataset...')

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

    logger.success('Done.')

def load_config_file(path: Path = CONFIG_FILE):
    """
    Loads the config file storing the data dictionary.

    Params:
        path (Path): Path to config file.
    
    Returns:
        config_file (dict): JSON in dictionary format.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config_file = json.load(f)

    return config_file

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
            df[column] = df[column].map({'Yes': 1, 'No': 0})
            encoded_cols.append(column)

    logger.success(f"Successfully encoded columns: {', '.join(encoded_cols)}")
    return df

def drop_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with a singular value.

    Params:
        df (DataFrame): Input dataframe.
    
    Returns:
        df (DataFrame): Dataframe with static columns dropped.
    """
    logger.info("Dropping columns with static values.")
    dropped = []
    for column in df.columns:
        if len(set(df[column].dropna().unique())) == 1:
            df = df.drop(labels=column, axis=1)
            dropped.append(column)

    logger.success(f"Successfully dropped columns: {', '.join(dropped)}")
    return df

def convert_dtypes(df: pd.DataFrame, dtype_map: dict = load_config_file()):
    """
    Converts columns in df to correct datatypes.

    Params:
        df (DataFrame): Unprocessed dataframe.
        dtype_map (dict): Mapping of columns to datatypes.
    
    Returns:
        df (DataFrame): Processed dataframe.
    """
    logger.info("Converting column datatypes.")
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if dtype.get('type') == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce', format=dtype.get('format'))
            elif dtype.get('type') == 'categorical':
                df[col] = df[col].astype(dtype='category')
            elif dtype.get('type') == 'ordinal':
                df[col] = df[col].astype(pd.CategoricalDtype(
                    categories=dtype.get('categories'),
                    ordered=dtype.get('ordered'))
                )

    logger.success("Converted column datatypes.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning steps on input DataFrame.

    Params:
        df (DataFrame): Input DataFrame.
    
    Returns:
        df (DataFrame): Processed DataFrame.
    """
    df = convert_dtypes(df=df, dtype_map=load_config_file())
    df = remove_outliers(df, domain_rules=DOMAIN_RULES) # Remove outliers
    df = encode_yes_no_cols(df) # Encode yes/no columns
    df = drop_static(df) # Drop static columns

    return df

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file as a pandas DataFrame
    
    Params:
        filepath (str): String filepath to the CSV file

    Returns:
        DataFrame: CSV file as a pandas DataFrame
    """
    return pd.read_csv(filepath)

def calculate_lower_upper(df: pd.DataFrame, col: str, iqr_multiplier: float = 1.5):
    """
    Calculate lower and upper bound for outlier detection.

    Params:
        df (DataFrame): DataFrame where outliers are being detected..
        col (str): Column to check for outliers.
        iqr_multiplier (float): Factor to multiply iqr by to set upper and lower bounds.
    
    Returns:
        lower_bound, upper_bound (tuple)
    """
    # Calculate iqr
    q1 = df[col].quantile(q=0.25)
    q3 = df[col].quantile(q=0.75)
    iqr = q3 - q1

    # Calculate lower and upper bounds
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)

    return (lower_bound, upper_bound)

def remove_outliers(df: pd.DataFrame, iqr_multiplier: float = 1.5, domain_rules: dict = None):
    """
    Removes outliers from numeric columsn using iqr.

    Params:
        df (DataFrame): Input dataframe
        iqr_multiplier (float): Factor to multiply iqr by in order to get lower and
            upper bounds.
    
    Returns:
        df (DataFrame): Processed dataframe with numeric outliers removed.
    """
    logger.info("Removing outliers.")
    # Get numeric columns
    numeric_columns = df.select_dtypes(include='number').columns

    original_length = len(df)
    # Iterate through numeric columns
    for col in numeric_columns:
        if col in domain_rules:
            # Retrieve min and max from domain_rules
            lower_bound = domain_rules[col].get('min')
            upper_bound = domain_rules[col].get('max')
        else:
            lower_bound, upper_bound = calculate_lower_upper(
                df=df,
                col=col,
                iqr_multiplier=iqr_multiplier
            )

        # Only select rows where values are within lower and upper bounds or are null
        df = df[(df[col].isna()) | ((df[col] >= lower_bound) & (df[col] <= upper_bound))]

    logger.success(f"Removed {original_length - len(df)} outliers.")
    return df

@app.command()
def main(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / "mxmh_survey_results.csv",
        help='Path to the raw dataset CSV'),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "processed_dataset.csv",
        help='Where to save the processed dataset')
):
    download_data() # Download CSV from Kaggle
    df = load_data(input_path) # Load raw CSV
    df = clean_data(df) # Perform cleaning
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    app()
