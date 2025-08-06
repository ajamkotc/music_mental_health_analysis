"""
Generates and saves a parameter grid for Gradient Boosting Classifier hyperparameter tuning.

This script creates a dictionary containing possible values for 'learning_rate', 'n_estimators',
and 'max_depth' to be used in hyperparameter search. The parameter grid is saved as a JSON file
to the path specified by `GBC_PARAM_DISTRIBUTION_FILE`.

Imports:
    json: For saving the parameter grid as a JSON file.
    numpy: For generating parameter value ranges.
    GBC_PARAM_DISTRIBUTION_FILE: File path for saving the parameter grid.

Parameter Grid:
    - learning_rate: 10 values linearly spaced between 0.001 and 0.2.
    - n_estimators: Integer values from 100 to 159 (inclusive).
    - max_depth: Integer values from 3 to 9 (inclusive).

Outputs:
    Writes the parameter grid to the specified JSON file and prints confirmation messages.
"""
import json
import numpy as np
from loguru import logger

from music_and_mental_health_survey_analysis.config import GBC_PARAM_DISTRIBUTION_FILE

# Define parameter distribution for GradientBoostingClassifier
rf_param_dist = {
    'learning_rate': [float(n) for n in np.linspace(start=0.001, stop=0.2, num=10)],
    'n_estimators': [int(n) for n in np.arange(start=100, stop=160, step=1)],
    'max_depth': [None] + [int(n) for n in np.arange(start=3, stop=10, step=1)]
}

# Save distribution to config file
with open(GBC_PARAM_DISTRIBUTION_FILE, 'w', encoding='utf-8') as f:
    json.dump(rf_param_dist, f, indent=2)

logger.success('Generate hyperparameter distribution for GradientBoostingClassifier.')
