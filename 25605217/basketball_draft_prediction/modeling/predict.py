"""Model prediction utilities."""

import pandas as pd
import numpy as np
from typing import Union, Tuple
import joblib

def make_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using trained model."""
    return model.predict(X)

def make_predictions_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Make probability predictions using trained model."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)

def create_submission(
    model,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """Create submission file for competition."""
    predictions = make_predictions(model, X_test)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'drafted': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission

def ensemble_predictions(
    models: list,
    X: pd.DataFrame,
    weights: list = None,
    method: str = 'voting'
) -> np.ndarray:
    """Create ensemble predictions from multiple models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if method == 'voting':
        predictions = np.zeros((len(X), len(models)))
        for i, model in enumerate(models):
            predictions[:, i] = make_predictions(model, X)
        
        ensemble_pred = np.round(np.average(predictions, axis=1, weights=weights))
    
    elif method == 'proba_averaging':
        probas = np.zeros((len(X), len(models)))
        for i, model in enumerate(models):
            probas[:, i] = make_predictions_proba(model, X)
        
        ensemble_proba = np.average(probas, axis=1, weights=weights)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred.astype(int)