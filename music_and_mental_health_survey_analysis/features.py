"""
Feature Engineering Module for Music and Mental Health Survey Analysis
This module provides functions for feature engineering on survey data, including
dummy encoding, binary conversion, and ordinal encoding of columns. It also
includes a CLI command to process a cleaned dataset and output engineered features.

Functions:
    - is_clean(df): Checks if the DataFrame has missing values.
    - dummy_encode(df, input_col, **kwargs): One-hot encodes a specified column.
    - col_to_binary(df, input_col, output_col=None, **kwargs): Converts a column to binary values.
    - ordinal_encode(df, input_col, **kwargs): Encodes ordinal features to numeric values.
    - apply_transformations(df, transform_functions=None):
        Applies configured transformations to the DataFrame.

CLI:
    - main(input_path, output_path): Loads cleaned data, applies feature engineering,
        and saves the result.

Configuration:
    - Uses external configuration files for feature transformation and datatype mapping.

Dependencies:
    - pandas, numpy, loguru, typer, warnings, pathlib
    - Project-specific modules: config, cleaning

Usage:
    Run as a script or import functions for use in other modules.

"""
import warnings
import json
from pathlib import Path

from loguru import logger
import typer

# Data manipulation
import pandas as pd
import numpy as np

# Project files and directories
from music_and_mental_health_survey_analysis.config import (
    PROCESSED_DATA_DIR, FEATURE_CONFIG_FILE, CONFIG_FILE, CONFIG_DIR
)
from music_and_mental_health_survey_analysis.cleaning import convert_dtypes
from music_and_mental_health_survey_analysis.utils import load_config_file, load_csv

app = typer.Typer()

def is_clean(df: pd.DataFrame):
    """
    Verifies that df is cleaned.

    Checks if there are any missing values in df.

    Params:
        df (DataFrame): Dataframe to check.

    Returns:
        bool: True if clean, False otherwise.
    """
    return any(df.isna())

def dummy_encode(df: pd.DataFrame, input_col: str, **kwargs) -> pd.DataFrame:
    """
    One-hot encodes input_col.

    Uses pd.get_dummies to one-hot encode the specified column in df.
    Accepts dtype kwarg but defaults to float type. If a prefix is not
    specified in kwargs then defaults to column name.

    Params:
        df (DataFrame): Dataframe to be processed.
        input_col (str): Column to be encoded.
        **kwargs (dict): Arbitrary keyword arguments.
            The following keywords are accepted:
                dtype (str): Datatype for encoded columns.
                prefix (str): Prefix for dummy columns.
    
    Returns:
        df (DataFrame): Processed dataframe.
    """
    logger.info(f"One-hot encoding column {input_col}")
    params = kwargs.get('params') # Get params if available

    # Verify input_col exists in df
    if input_col in df.columns:
        # Get dummy variables
        dummies = pd.get_dummies(
            df[input_col],
            dtype=float,
            prefix=params.get('prefix', input_col),
            drop_first=True)

        # Combine and drop original column
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(labels=input_col, axis=1)
    else:
        # Input_col does not exist in df
        warnings.warn(
            message=f"{input_col} not in dataframe.",
            category=UserWarning
        )

    logger.success("Successfully encoded.")
    return df

def append_dtypes_json(col: str, dtype: str, file_path: Path = CONFIG_FILE, **kwargs):
    """
    Appends a new datatype entry to the dtype config.

    Checks if the column is already listed in the config file. If not, generates new entry based
    on dtype and arbitrary keyword arguments, then writes it to a .json file.

    Params:
        col (str): New column to add to config
        dtype (str): Datatype of new column
        file_path (Path): Path to config file
        **kwargs (dict): Arbitrary keyword argument.
            Accepted keyword arguments:
                - categories (list): List of categories for category type
                - ordered (bool): Whether categories are ordered or not
    """
    dtype_file = load_config_file(file_path) # Loads dtype config file
    new_entry = {'type': dtype} # Initializes base entry

    # Checks if entry already exists in config file
    if not col in dtype_file.items():
        # Gets arbitrary keyword arguments
        categories = kwargs.get('categories')
        ordered = kwargs.get('ordered')

        if categories:
            # If list of categories included
            new_entry['categories'] = categories
        if ordered:
            # If ordered specification included
            new_entry['ordered'] = ordered

        # Update dtype dictionary
        dtype_file.update({col: new_entry})

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dtype_file, f, indent=4)

def col_to_binary(
        df: pd.DataFrame,
        input_col: str,
        output_col: str = None,
        **kwargs) -> pd.DataFrame:
    """
    Convert specified column to contain binary values.

    A new column is created as the binary representation of the inputted column.
    If a positive class is specified then that is used to apply the transformation,
    otherwise dummy cols are created instead.

    Params:
        df (DataFrame): Input dataframe.
        col (str): Column to be converted to binary.
        output_col (str): New binary column.
        **kwargs (dict): Arbitrary keyword arguments.
            The following keywords are accepted:
                positive_class (str): Class to act as positive class during transformation.

    Returns:
        df (DataFrame): Processed dataframe with transformed column.
    """
    logger.info(f"Binary encoding column {input_col}")

    # If no output column name specified, set to same as col
    if output_col is None:
        output_col = input_col

    # Check if column in DataFrame
    if input_col in df.columns:
        params = kwargs.get('params')

        # Check if a positive class has been identified
        if params.get('positive_class'):
            # Convert to binary represented by 0s and 1s
            df[output_col] = (
                df[input_col] == params.get('positive_class')).astype(float)

            # Drop original column if different from output_col
            if input_col != output_col:
                df = df.drop(labels=input_col, axis=1)
                append_dtypes_json(
                    col=output_col,
                    dtype='binary',
                    file_path=CONFIG_DIR / 'column_types_test.json'
                )

        else:
            # If no positive class identified, return with dummy cols
            warnings.warn(
                message='No positive class identified for binary conversion.',
                category=UserWarning)

            dummies = pd.get_dummies(df[input_col], dtype=int, prefix=input_col)
            df = pd.concat([df.drop(input_col, axis=1), dummies], axis=1)

    logger.success("Successfully encoded.")
    return df

def ordinal_encode(df: pd.DataFrame, input_col: str, **kwargs) -> pd.DataFrame:
    """
    Encodes ordinal features to numeric.

    Issues a warning if input_col is not in the df.
    When provided, applies mapping to convert ordinal features to numeric. If
    a dtype is provided converts to that type, otherwise defaults to float.

    Params:
        df (DataFrame): Dataframe to process.
        input_col (str or list): Column(s) to encode.
        **kwargs (dict): Optional keyword arguments.
            The following keywords are accepted:
                mapping (dict): Dictionary mapping values to numeric representations.
                dtype (str): Dtype to convert new mapping to. Defaults to float.
    
    Returns:
        df (DataFrame): Processed dataframe.
    """
    logger.info(f"Ordinal encoding column(s) {input_col}")
    params = kwargs.get('params') # Access transformation params

    # Check if input_col is a list of columns or single column name
    if isinstance(input_col, list):
        # input_col is a list
        for col in input_col:
            # Double check that column is in the dataframe
            if col in df.columns:
                if params.get('mapping'):
                    # Apply mapping using mapping params
                    df[col] = df[col].map(params.get('mapping')).astype(params.get('dtype', float))
                else:
                    # Apply mapping using category codes
                    df[col] = df[col].cat.codes.replace(-1, np.nan)
                    df[col] = df[col].astype(params.get('dtype', float))
            else:
                # input_col was not found in df
                warnings.warn(
                    message=f"Column {col} was not found in the dataframe.",
                    category=UserWarning
                )
    else:
        # input_col is a single column
        if params.get('mapping'):
            df[col] = df[col].map(params.get('mapping')).astype(params.get('dtype', float))
        else:
            warnings.warn(
                message=f"No mapping specified. {input_col} could not be encoded.",
                category=UserWarning
            )

    logger.success("Successfully encoded")
    return df

def apply_transformations(
        df: pd.DataFrame,
        transform_functions: dict = None) -> pd.DataFrame:
    """
    Applies transformations to dataframe.

    Checks features_config.json for transformations to perform. Looks up the
    transformations in transform_functions and passes the necessary parameters
    to the function.

    Params:
        df (DataFrame): Dataframe to process.
        transform_functions (dict): Dictionary of transform function mappings.

    Returns:
        df (DataFrame): Dataframe with applied transformations.
    """
    if not is_clean(df=df):
        warnings.warn(
            message='Dataframe is not clean. Anomalies may occur when transforming columns.',
            category=UserWarning
        )

    logger.info("Applying column transformations.")

    feature_config = load_config_file(FEATURE_CONFIG_FILE)
    # Sets to default if none is passed
    if transform_functions is None:
        transform_functions = TRANSFORM_FUNCTIONS

    # Loop through transformations in feature_config
    for _, cfg in feature_config.items():
        func = transform_functions.get(cfg['transform'])
        if func:
            df = func(
                df=df,
                input_col=cfg.get('input_column'),
                output_col=cfg.get('output_column', None),
                params=cfg.get('params', {})
            )

    logger.success("Completed column transformations.")
    return df

TRANSFORM_FUNCTIONS = {
    'col_to_binary': col_to_binary,
    'ordinal_encode': ordinal_encode,
    'dummy_encode': dummy_encode
}

@app.command()
def main(
    input_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "cleaned.csv",
        help='Cleaned dataset to be used for feature engineering.'
    ),
    output_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "features.csv",
        help='Output path for engineered dataset.'
    )
):
    """
    Performs feature engineering on a cleaned dataset and saves the resulting features
        to a specified output path.
    Args:
        input_path (Path, optional): Path to the cleaned input CSV file.
            Defaults to PROCESSED_DATA_DIR / "cleaned.csv".
        output_path (Path, optional): Path to save the engineered features CSV file.
            Defaults to PROCESSED_DATA_DIR / "features.csv".

    Workflow:
        1. Loads the cleaned dataset from the specified input path.
        2. Loads datatype configuration and applies it to the DataFrame.
        3. Applies feature transformations to the DataFrame.
        4. Saves the engineered features to the specified output path.
    """
    logger.info('Beginning feature engineering.')
    df = load_csv(input_path) # Load cleaned data

    # Set datatypes since reset when load csv
    dtype_config = load_config_file(CONFIG_FILE)
    df = convert_dtypes(df=df, dtype_map=dtype_config)

    df = apply_transformations(df=df) # Apply column transformations

    logger.success('Completed feature engineering.')
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    app()
