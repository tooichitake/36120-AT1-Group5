"""
Basketball Draft Prediction - Modeling Module

This module provides model training and prediction utilities.
"""

from . import train
from . import predict

# Expose key classes and functions for convenience
from .train import (
    LightGBMTrainer,
    CatBoostTrainer,
    RandomForestTrainer,
    get_lightgbm_base_params,
    get_catboost_base_params,
    get_randomforest_base_params
)

from .predict import (
    predict_with_model,
    load_and_predict,
    predict_lightgbm,
    predict_catboost,
    predict_sklearn,
    create_submission,
    ensemble_predictions,
    evaluate_model_performance,
    business_impact_analysis
)

__all__ = [
    # Modules
    'train',
    'predict',
    
    # Trainer classes
    'LightGBMTrainer',
    'CatBoostTrainer', 
    'RandomForestTrainer',
    
    # Training functions
    'get_lightgbm_base_params',
    'get_catboost_base_params',
    'get_randomforest_base_params',
    
    # Prediction functions
    'predict_with_model',
    'load_and_predict',
    'predict_lightgbm',
    'predict_catboost',
    'predict_sklearn',
    'create_submission',
    'ensemble_predictions',
    'evaluate_model_performance',
    'business_impact_analysis'
]