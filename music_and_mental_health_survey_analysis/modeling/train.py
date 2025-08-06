import subprocess
from pathlib import Path
import pickle
from datetime import datetime

from loguru import logger
import typer

# Data manipulation
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate
)

# Models
from sklearn.ensemble import GradientBoostingClassifier

# Evaluating
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from music_and_mental_health_survey_analysis.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, GBC_PARAM_DISTRIBUTION_FILE, GBC_MODEL_CONFIG_FILE, REPORTS_DIR
)
from music_and_mental_health_survey_analysis.utils import load_config_file

app = typer.Typer()

TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_SPLITS = 10
SCORING = {
    'accuracy': 'accuracy',
    'precision': make_scorer(score_func=precision_score, average='binary', zero_division=0),
    'recall': make_scorer(score_func=recall_score, average='binary', zero_division=0),
    'f1': make_scorer(score_func=f1_score, average='binary', zero_division=0),
    'roc_auc': 'roc_auc'
}

def tune(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        kf: StratifiedKFold,
        scoring: str = 'roc_auc') -> GradientBoostingClassifier:
    """
    Tune GradientBoostingClassifier using RandomizedSearchCV

    Takes in X and y training data, a cross-validation object, and scoring method to conduct
    a randomized search. If the parameter distribution file has not already been generated it does
    so using generate_param_grids.py script.

    Params:
        X_train (DataFrame): Feature training data.
        y_train (Series): Labels training data.
        kf (StratifiedKFold): KFold object for cross-validation.
        scoring (str): Scoring method for hyperparameter search.
    
    Returns:
        GradientBoostingClassifier: Best estimator found by the randomized search.
    """
    param_file = Path(GBC_PARAM_DISTRIBUTION_FILE)

    # If param distribution file doesn't exist, generate using generate_param_grids.py script
    if not param_file.exists():
        logger.info(f"{param_file} not found. Generating it now...")
        subprocess.run(["python", "config/generate_param_grids.py"], check=True)

    # Load param distribution from config file
    param_dist = load_config_file(path=param_file)

    random_search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_jobs=-1,
        cv=kf,
        scoring=scoring,
        random_state=RANDOM_STATE,
        verbose=1
    )

    logger.info("Beginning hyperparameter tuning with RandomizedSearchCV...")
    random_search.fit(X=X_train, y=y_train)

    logger.success(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

def evaluate_train(estimator, X_train: pd.DataFrame, y_train: pd.Series, cv, scoring: dict) -> dict:
    """
    Evaluate model performance on training data.

    Perform 
    """
    logger.info('Evaluating training metrics.')
    cv_results = cross_validate(
        estimator=estimator,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring
    )

    results = {'timestamp': datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}
    scores = {score: round(np.mean(cv_results[f"test_{score}"]), 3) for score in scoring}
    results.update(scores)

    logger.success(f"Evaluated: {', '.join(list(scoring.keys()))}")

    return results

def pickle_model(model, path: Path) -> None:
    """
    Saves a model as a pickled object.

    Params:
        model: Model object.
        path (Path): Path to save model.
    """
    logger.info('Saving model.')

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    logger.success(f"Saved model to {path}")

def init_model(tune_model: bool, X_train: pd.DataFrame, y_train: pd.Series, cv, param_path: Path):
    """
    Initializes a gradient boosting model.

    If the model should be tuned, it will pass the data to the tune function. Otherwise it will
    initialize a GradientBoostingClassifier and fit it to the data.

    Params:
        tune_model (bool): Whether hyperparameters should be tuned.
        X_train (DataFrame): Training feature set.
        y_train (Series): Training label set.
        cv (KFold): Cross-validation object.
        param_path (Path): Path to model parameter .json file
    
    Returns:
        model (GradientBoostingClassifier): Tuned and fitted model.
    """
    if tune_model:
        model = tune(X_train=X_train, y_train=y_train, kf=cv) # Get best model
    else:
        logger.info('Initializing GradientBoostingClassifier model.')

        model_config = load_config_file(path=param_path) # Load model hyperparameters
        model = GradientBoostingClassifier(**model_config)
        model.fit(X_train, y_train) # train model

        logger.success(f"Initialized model with parameters {model_config}")

    return model

def save_metrics(metrics: dict, output_path: Path) -> None:
    """
    Saves metrics to a csv file.

    If the file already exists, it appends the metrics to the file. Otherwise it creates a new one.

    Params:
        metrics (dict): Dictionary of metrics and their scores.
        output_path (Path): Path to save .csv file.
    """
    metrics_df = pd.DataFrame([metrics])

    if output_path.exists():
        metrics_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_path, index=False)

    logger.success(f"Saved metrics to {output_path}")

@app.command()
def main(
    features_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "features.csv",
        help='Path to features dataset.'
    ),
    labels_path: Path = typer.Option(
        default=PROCESSED_DATA_DIR / "labels.csv",
        help='Path to labels dataset.'
    ),
    model_path: Path = typer.Option(
        default=MODELS_DIR / "gbc_model.pkl",
        help='Path to save pickled model.'
    ),
    tune_model: bool = typer.Option(
        default=False,
        help='Determines whether model should be tuned or not.'
    )
):
    """
    Trains a Gradient Boosting Classifier model on provided features and labels.

    Args:
        features_path (Path): Path to the CSV file containing feature data. 
            Defaults to 'features.csv' in the processed data directory.
        labels_path (Path): Path to the CSV file containing label data. 
            Defaults to 'labels.csv' in the processed data directory.
        model_path (Path): Path to save the trained and pickled model. 
            Defaults to 'gbc_model.pkl' in the models directory.
        tune_model (bool): If True, performs hyperparameter tuning before training the model.
            Defaults to False.

    Workflow:
        1. Loads feature and label datasets from the specified paths.
        2. Splits the data into training and test sets using stratified sampling.
        3. Sets up stratified K-fold cross-validation.
        4. Initializes and trains a Gradient Boosting Classifier, optionally tuning hyperparameters.
        5. Saves the trained model to the specified path.

    Logs progress and key steps throughout the process.
    """
    # Load data
    logger.info('Loading features and labels.')
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    logger.success('Data loaded.')

    # Split data
    logger.info('Splitting data into training and test sets.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE, stratify=y
    )
    logger.success('Data split.')

    # Set up cross-validation
    logger.info(f"Setting up {CV_SPLITS}-fold cross-validation.")
    kf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    logger.success("Cross-Validation set up.")

    # Initialize model
    model = init_model(
        tune_model=tune_model,
        X_train=X_train,
        y_train=y_train,
        cv=kf,
        param_path=GBC_MODEL_CONFIG_FILE
    )

    pickle_model(model=model, path=model_path) # Pickle and save model

    # Evaluate training performance and save
    train_scores = evaluate_train(
        estimator=model,
        X_train=X_train,
        y_train=y_train,
        cv=kf,
        scoring=SCORING
    )
    save_metrics(metrics=train_scores, output_path=REPORTS_DIR / 'training_scores.csv')

if __name__ == "__main__":
    app()
