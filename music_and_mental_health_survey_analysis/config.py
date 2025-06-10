import json
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Data manipulation
import pandas as pd

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLEANING_EVALUATIONS_DIR = FIGURES_DIR / "cleaning_evaluations"
SAMPLING_EVALUATIONS_DIR = FIGURES_DIR / "sampling_evaluations"

KAGGLE_DATASET = 'catherinerasgaitis/mxmh-survey-results'

# Config files
CONFIG_DIR = PROJ_ROOT / "config"

CONFIG_FILE = CONFIG_DIR / "column_types.json"
DOMAIN_RULES = CONFIG_DIR / "domain_rules.json"
FEATURE_CONFIG_FILE = CONFIG_DIR / "feature_config.json"
SAMPLING_CONFIG_FILE = CONFIG_DIR / "sampling_config.json"

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

def load_csv(path: Path) -> pd.DataFrame:
    """
    Loads data from CSV into pandas DataFrame.

    Params:
        path (Path): Path to CSV file.
    
    Returns:
        DataFrame: CSV loaded into DataFrame.
    """
    return pd.read_csv(path)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
