"""
This module provides functions and a CLI application for cleaning and 
preprocessing survey data related to music and mental health.

Main functionalities include:
- Removing outliers from numeric columns using IQR or domain-specific rules.
- Dropping timestamp columns and columns with static (single) values.
- Identifying continuous (numeric) and categorical columns based on a configuration file.
- Converting dataframe columns to appropriate datatypes as specified in a configuration file.
- Imputing missing values using KNN for numeric columns and the most frequent value for     
    categorical columns.
- Providing a Typer CLI command to perform the full cleaning pipeline, including loading raw data, 
    applying all cleaning steps, and saving the cleaned data.

Functions:
    - calculate_lower_upper: Calculates lower and upper bounds for outlier detection using IQR.
    - remove_outliers: Removes outliers from numeric columns using IQR or domain rules.
    - drop_timestamp: Drops columns with datetime types.
    - drop_static: Drops columns with only a single unique value.
    - list_continuous_categorical: Separates columns into continuous and categorical 
        based on a config.
    - convert_dtypes: Converts dataframe columns to specified datatypes.
    - impute_missing: Imputes missing values in the dataframe.
    - clean: CLI command to perform all cleaning steps and save the result.
    - main: CLI entry point.
    
Usage:
    Run as a CLI tool to clean a dataset:
        python cleaning.py clean --input-path <input_csv> --output-path <output_csv>
"""
from pathlib import Path

import warnings
import typer

from loguru import logger

# Data manipulation
import pandas as pd
import numpy as np

# Imputing
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

# Project files and directors
from music_and_mental_health_survey_analysis.config import (
    DOMAIN_RULES, RAW_DATA_DIR, PROCESSED_DATA_DIR, CONFIG_FILE
)

from music_and_mental_health_survey_analysis.utils import load_config_file, load_csv

app = typer.Typer()

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

def remove_outliers(
        df: pd.DataFrame,
        iqr_multiplier: float = 1.5,
        domain_rules: dict = DOMAIN_RULES):
    """
    Removes outliers from numeric columns using iqr or domain rules.

    Iterates through columns of a dataframe to remove outliers. If a column is found
    in domain rules, then it uses the specified rules to remove outliers. This applies to
    features such as Age where iqr is not appropriate for detecting outliers.

    For other features, it calculates lower and upper bounds using the iqr_multiplier
    multiplied by the iqr. These columns are dropped.

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
            upper_bound = domain_rules[col].get('max', np.inf)
        else:
            lower_bound, upper_bound = calculate_lower_upper(
                df=df,
                col=col,
                iqr_multiplier=iqr_multiplier
            )

        # Only select rows where values are within lower and upper bounds or are null
        df = df[(df[col].isna()) | ((df[col] > lower_bound) & (df[col] <= upper_bound))]

    logger.success(f"Removed {original_length - len(df)} outliers.")
    return df

def drop_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop timestamp columns.

    Iterates through the inputted dataframe and uses is_datetime64_any_dtype
    from pandas.api.types library to detect datetime columns to drop.

    Params:
        df (DataFrame): Dataframe to process.
    
    Returns:
        df (DataFrame): Dataframe with timestamp columns removed.
    """
    logger.info("Dropping timestamp columns.")
    count = 0

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.drop(col, axis=1)
            count += 1

    logger.success(f"Successfully dropped {count} columns.")
    return df

def drop_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with a singular value.

    Iterates through each column of the inputted dataframe and checks the 
    number of unique stored values. If equal to 1 then that column is dropped.

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

def list_continuous_categorical(
        df: pd.DataFrame,
        dtype_dict: dict = load_config_file(CONFIG_FILE),
        encoded: bool = False) -> tuple:
    """
    Generates a tuple of continuous and categorical columns.

    Loops though datatype dictionary and assigns columns to either categorical_cols
    or numeric_cols based on the "type" feature. Any columns which are skipped
    are saved to issue a warning since this reflects changes in the schema prior to
    imputation.

    If encoded is set to True, then column names are split at the underscore since that is how
    dummy columns are denoted.

    Params:
        df (DataFrame): Dataframe to parse through.
        dtype_dict (dict): Mapping of columns to dtypes.
        encoded (bool): Whether df is encoded or not.

    Returns:
        numeric_cols, categorical_cols (tuple): Tuple separating numeric and categorical columns.
    """
    numeric_cols = []
    categorical_cols = []
    skipped_cols = []

    # Iterate through df columns
    for col in df.columns:
        # Get datatype information
        dtype = dtype_dict.get(col)

        if dtype:
            # If dtype info is found for col
            if dtype.get('type') == 'numeric':
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        elif encoded:
            # If encoded split at underscore to get original column name
            dtype = dtype_dict.get(col.split('_')[0])

            if dtype:
                if dtype.get('type') == 'numeric':
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        else:
            skipped_cols.append(col)

    # Warn user in case columns were not found in df
    if skipped_cols:
        warnings.warn(
            "The following columns in dtype_dict were not found in the DataFrame " \
            f"and were skipped: {', '.join(skipped_cols)}",
            category=UserWarning
        )

    return (numeric_cols, categorical_cols)

def convert_dtypes(df: pd.DataFrame, dtype_map: dict = load_config_file(CONFIG_FILE)):
    """
    Converts dataframe columns 

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
            elif dtype.get('type') == 'categorical' or dtype.get('type') == 'binary':
                df[col] = df[col].astype(dtype='category')
            elif dtype.get('type') == 'ordinal':
                df[col] = pd.Categorical(
                    df[col],
                    categories=dtype.get('categories'),
                    ordered=dtype.get('ordered')
                )
            elif dtype.get('type') == 'numeric' or dtype.get('type') == 'binary':
                df[col] = df[col].astype(float)

    logger.success("Converted column datatypes.")
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in df.

    Uses KNNImputer to impute continuous features and SimpleImputer with the most_frequent
    strategy to impute categorical features.

    Params:
        df (DataFrame): Dataframe with missing values.
        dtype_dict (dict): Dictionary mapping columns to dtypes.

    Returns:
        df (DataFrame): Dataframe with imputed missing values.
    """
    logger.info("Imputing missing values.")
    dtype_dict = load_config_file(CONFIG_FILE)
    numeric_cols, cat_cols = list_continuous_categorical(df=df, dtype_dict=dtype_dict)

    processor = ColumnTransformer(
        transformers=[
            ('num', KNNImputer(), numeric_cols),
            ('cat', SimpleImputer(strategy='most_frequent'), cat_cols)
        ]
    )

    imputed_data = processor.fit_transform(df)
    logger.success("Imputed missing values.")
    return pd.DataFrame(imputed_data, columns=numeric_cols+cat_cols)

@app.command()
def clean(
    input_path: Path = typer.Option(
        default=RAW_DATA_DIR / 'mxmh_survey_results.csv',
        help='Path to dataset to be cleaned.'
    ),
    output_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / 'cleaned.csv',
        help='Path to location to save cleaned dataset.'
    )
):
    """
    Perform data cleaning steps.

    Steps:
        1. Drop static columns
        2. Convert columns to correct datatypes
        3. Drop timestamp columns
        4. Remove outliers
        5. Impute missing values
    """
    logger.info("Beginning dataset cleaning.")
    df = load_csv(path=input_path) # Load data
    df = drop_static(df=df) # Drop static columns

    # Convert datatypes
    dtype_config_file = load_config_file(path=CONFIG_FILE)
    df = convert_dtypes(df=df, dtype_map=dtype_config_file)

    df = drop_timestamp(df=df) # Drop timestamp columns

    # Remove outliers
    domain_rules = load_config_file(path=DOMAIN_RULES)
    df = remove_outliers(df=df, iqr_multiplier=1.5, domain_rules=domain_rules)

    df = impute_missing(df=df) # Impute missing values

    logger.success("Successfully cleaned dataset.")
    df.to_csv(output_path, index=False) # Save dataframe

if __name__ == '__main__':
    app()
