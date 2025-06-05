import warnings
from pathlib import Path

from loguru import logger
import typer

# Data manipulation
import pandas as pd
import numpy as np

# Modeling
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

from music_and_mental_health_survey_analysis.config import (
    PROCESSED_DATA_DIR, CONFIG_FILE
)
from music_and_mental_health_survey_analysis.dataset import (
    convert_dtypes, load_config_file
)

app = typer.Typer()

def load_csv(path: Path) -> pd.DataFrame:
    """
    Loads data from CSV into pandas DataFrame.

    Params:
        path (Path): Path to CSV file.
    
    Returns:
        DataFrame: CSV loaded into DataFrame.
    """
    return pd.read_csv(path)

def encode(df: pd.DataFrame, dtype_dict: dict = load_config_file()) -> pd.DataFrame:
    """
    Encode features to floats.

    Params:
        df (DataFrame): Input dataframe to process.
        dtype_dict (dict): Mapping dictionary of dtypes.

    Returns:
        df (DataFrame): Processed dataframe.
    """
    logger.info("Encoding features.")
    for col, dtype in dtype_dict.items():
        if col in df.columns:
            if dtype.get('type') == 'categorical':
                # One hot encode categorical columns
                df = pd.get_dummies(data=df, columns=[col], dtype='float')
            elif dtype.get('type') == 'ordinal':
                # Ordinal columns with ordering
                df[col] = df[col].cat.codes.replace(-1, np.nan)
                df[col] = df[col].astype(float)

    logger.success("Encoded all features.")
    return df

def list_continuous_categorical(df: pd.DataFrame, dtype_dict: dict = load_config_file()) -> tuple:
    """
    Generates a tuple of continuous and categorical columns.

    Params:
        df (DataFrame): Dataframe to parse through.
        dtype_dict (dict): Mapping of columns to dtypes.

    Returns:
        numeric_cols, categorical_cols (tuple): Tuple separating numeric and categorical columns.
    """
    numeric_cols = []
    categorical_cols = []
    skipped_cols = []

    for col, dtype in dtype_dict.items():
        if col in df.columns:
            if dtype.get('type') == 'numeric':
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            # Keep track of columns listed in dtype_dict but not in df
            skipped_cols.append(col)

    # Warn user in case columns were not found in df
    if skipped_cols:
        warnings.warn(
            "The following columns in dtype_dict were not found in the DataFrame " \
            f"and were skipped: {', '.join(skipped_cols)}",
            category=UserWarning
        )

    return (numeric_cols, categorical_cols)

def impute_missing(df: pd.DataFrame, dtype_dict: dict = load_config_file()) -> pd.DataFrame:
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
def main(
    input_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "processed_dataset.csv",
        help='Cleaned dataset to be used for feature engineering.'
    ),
    output_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "features.csv",
        help='Output path for engineered dataset.'
    )
):
    df = load_csv(input_path)
    df = convert_dtypes(df=df, dtype_map=load_config_file(CONFIG_FILE))
    df = impute_missing(df=df)
    #df = encode(df)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    app()
