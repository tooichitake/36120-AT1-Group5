"""
Basketball Draft Prediction Package

A comprehensive package for predicting NBA draft prospects using machine learning.
"""

# Import main modules for easier access
from . import dataset
from . import features
from . import visualization
from . import modeling

# Import submodules from modeling for direct access
from .modeling import train, predict

# Version information
__version__ = "1.0.0"
__author__ = "Zhiyuan Zhao"
__email__ = "25605217@student.uts.edu.au"

# Package metadata
__all__ = [
    "dataset",
    "features", 
    "visualization",
    "modeling",
    "train",
    "predict"
]

# Convenience imports for commonly used functions
from .dataset import (
    load_data,
    split_data_stratified,
    convert_to_float64,
    convert_height_to_inches,
    create_target_variable
)

from .features import (
    create_power_conference_indicator,
    apply_all_feature_engineering,
    lightgbm_feature_importance_analysis,
    catboost_feature_importance_analysis,
    randomforest_feature_importance_analysis
)

from .visualization import (
    set_visualization_style,
    plot_target_distribution,
    plot_feature_correlations,
    analyze_feature_by_draft_status,
    evaluate_model
)