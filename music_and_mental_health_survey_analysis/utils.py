"""
Utility functions for loading configuration files and CSV data.

Functions:
    load_config_file(path: Path) -> dict:
        Loads a JSON configuration file from the specified path and returns its contents as a dictionary.
    load_csv(path: Path) -> pd.DataFrame:
        Loads a CSV file from the specified path into a pandas DataFrame.
"""
import json
from pathlib import Path

import pandas as pd

def load_config_file(path: Path):
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

def load_csv(path: Path) -> pd.DataFrame:
    """
    Loads data from CSV into pandas DataFrame.

    Params:
        path (Path): Path to CSV file.
    
    Returns:
        DataFrame: CSV loaded into DataFrame.
    """
    return pd.read_csv(path)