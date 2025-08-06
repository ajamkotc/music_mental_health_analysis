
"""
This module provides functions and a CLI for balancing datasets through oversampling and
undersampling techniques, specifically tailored for use in music and mental health survey analysis.
It supports methods such as SMOTE-NC for oversampling and Tomek Links for undersampling,
and is configurable via external configuration files.

Functions:
    is_transformed(df: pd.DataFrame) -> bool:
        Checks if a DataFrame has been preprocessed by verifying data types and missing values.
    undersample(df: pd.DataFrame, target_column: str, method: str = 'tomek') -> pd.DataFrame:
        Applies undersampling to the majority class in the specified target column using
        the given method (default: Tomek Links).
    get_col_indeces(df: pd.DataFrame, cols: list) -> list:
        Returns the indices of specified columns in the DataFrame.
    oversample(df: pd.DataFrame, target_column: str, method: str = 'SMOTE-NC') -> pd.DataFrame:
        Oversamples the minority class in the target column using the specified method
        (default: SMOTE-NC).
    balance(df: pd.DataFrame, sampling_config: dict, sample_functions: dict = None) -> pd.DataFrame:
        Balances the dataset according to the provided sampling configuration
        and sampling functions.
    main(input_path: Path, output_path: Path):
        CLI entry point for balancing a dataset. Loads a transformed dataset,
        applies balancing techniques, and saves the balanced dataset to the
        specified output path.

Constants:
    SAMPLE_FUNCTIONS (dict): Mapping of sampling types ('undersample', 'oversample')
    to their respective functions.

Usage:
    This module can be run as a script to balance a dataset using Typer CLI, or its functions
    can be imported and used programmatically.
"""
import warnings
from pathlib import Path

from loguru import logger
import typer

# Data manipulation
import pandas as pd

# Balancing
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks

# Project files and directories
from music_and_mental_health_survey_analysis.config import (
    PROCESSED_DATA_DIR, SAMPLING_CONFIG_FILE
)
from music_and_mental_health_survey_analysis.cleaning import list_continuous_categorical
from music_and_mental_health_survey_analysis.utils import load_config_file, load_csv
from music_and_mental_health_survey_analysis.features import is_clean

app = typer.Typer()

def is_transformed(df: pd.DataFrame):
    """
    Checks if df has been preprocessed.

    Checks if any datatypes are numeric or if any values are missing.

    Params:
        df (DataFrame): Dataframe to be checked
    
    Returns:
        bool: True if transformed, False otherwise
    """
    return any(df.select_dtypes(exclude='number') or is_clean(df=df))

def undersample(df: pd.DataFrame, target_column: str, method: str = 'tomek') -> pd.DataFrame:
    """
    Applies undersampling to a dataframe.

    By default applies Tomek Links method of undersampling the target_column of the input df.
    If applied Tomeks then target_column must be a binary column, otherwise no undersampling
    will occur and user will receive a warning.

    Params:
        df (DataFrame): Input dataframe.
        target_column (str): Column in which to undersample majority class.
        method (str): Method of undersampling.
    
    Returns:
        df (DataFrame): Undersampled dataframe
    """
    logger.info(f"Undersampling {target_column} with {method}.")
    # Split data into features and target feature
    X = df.drop(labels=[target_column], axis=1)
    y = df[target_column]

    if method == 'tomek':
        # Verify that target feature is binary column before applying tomek
        if y.nunique() == 2:
            tl = TomekLinks()
            X_resampled, y_resampled = tl.fit_resample(X, y)

            # Combine resampled data into dataframe
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            df = pd.concat(
                [resampled_df, pd.DataFrame(y_resampled, columns=[target_column])],
                axis=1)
        else:
            # Target feature is not binary
            warnings.warn(
                message='Binary target feature required for Tomeks Links undersampling.',
                category=UserWarning
            )

    logger.success('Successfully undersampled.')
    return df

def get_col_indeces(df: pd.DataFrame, cols: list):
    """
    List indeces of input columns in df.

    Iterate through cols and get their indeces in df.

    Params:
        df (DataFrame): Dataframe to iterate through.
        cols (list): List of columns
    
    Returns:
        col_indeces (list): List of column indeces in df.
    """
    return df.columns.get_indexer(cols).tolist()

def oversample(df: pd.DataFrame, target_column: str, method: str = 'SMOTE-NC') -> pd.DataFrame:
    """
    Oversample minority class in target_column.

    Applies specified method of oversampling to df based on the target_column. The default
    method is SMOTE-NC.

    Params:
        df (DataFrame): Input dataframe.
        target_column (str): Column to oversample.
        method (str): Method of oversampling. Default is SMOTE-NC.
    
    Returns:
        df (DataFrame): Oversampled dataframe.
    """
    logger.info(f"Oversampling {target_column} using {method}.")
    # Split data into features and target variable
    X = df.drop(labels=[target_column], axis=1)
    y = df[target_column]

    if method == 'SMOTE-NC':
        _, categorical = list_continuous_categorical(df=X, encoded=True)
        cat_indeces = get_col_indeces(df=X, cols=categorical)

        smote_nc = SMOTENC(categorical_features=cat_indeces)

        # Apply
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        df = pd.concat([resampled_df, pd.DataFrame(y_resampled, columns=[target_column])], axis=1)

    logger.success("Successfully oversampled.")
    return df

def balance(
        df: pd.DataFrame,
        sampling_config: dict,
        sample_functions: dict = None) -> pd.DataFrame:
    """
    Balance imbalanced data.

    Applies balancing functions based on the sampling_config dictionary. Returns a processed
    dataframe.

    Params:
        df (DataFrame): Dataframe to be processed
        sampling_config (dict): Dictionary defining sampling to occur in df.
        sample_functions (dict): Dictionary mapping sample types defined in 
            sampling_config to functions.
    
    Returns:
        df (DataFrame): Balanced dataframe.
    """
    if sample_functions is None:
        sample_functions = SAMPLE_FUNCTIONS

    for sampling in sampling_config.get('balance_strategy'):
        func = sample_functions.get(sampling.get('type'))

        if func:
            df = func(
                df=df,
                target_column=sampling_config.get('target_column'),
                method=sampling.get('method')
            )

    return df

SAMPLE_FUNCTIONS = {
    'undersample': undersample,
    'oversample': oversample
}

@app.command()
def main(
    input_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "features.csv",
        help='Transformed dataset to be used for sampling.'
    ),
    output_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR,
        help='Output path for sampled dataset.'
    )
):
    """
    Main function to perform sampling on a transformed dataset.

    This function loads a pre-processed dataset, applies balancing techniques to address
    class imbalance based on a provided sampling configuration, and saves the resulting sampled
    dataset to the specified output path.

    Args:
        input_path (Path, optional): Path to the transformed dataset CSV file.
            Defaults to PROCESSED_DATA_DIR / "features.csv".
        output_path (Path, optional): Path to save the sampled dataset CSV file.
            Defaults to PROCESSED_DATA_DIR / "sampled.csv".

    Returns:
        None

    Side Effects:
        Writes the balanced and sampled dataset to the specified output path as a CSV file.
    """
    logger.info('Balancing dataset.')
    df = load_csv(input_path) # Load transformed data

    # Balance imbalanced target column
    sampling_config = load_config_file(SAMPLING_CONFIG_FILE)
    df = balance(df=df, sampling_config=sampling_config, sample_functions=SAMPLE_FUNCTIONS)

    logger.success('Data balanced.')

    # Split into features and labels to save

    logger.info(f"Saving features and labels to {output_path}")
    X = df.drop(labels=['improved'], axis=1)
    y = df['improved']

    X.to_csv(output_path / 'features.csv', index=False)
    y.to_csv(output_path / 'labels.csv', index=False)

    logger.success('Successfully saved data.')

if __name__ == '__main__':
    app()
