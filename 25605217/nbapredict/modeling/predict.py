"""Model prediction utilities."""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Any, Dict
import joblib
import lightgbm as lgb
import catboost as cb
from pathlib import Path

def predict_with_model(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with any trained model.
    
    Parameters:
    -----------
    model : Any
        Trained model (LightGBM, CatBoost, RandomForest, etc.)
    X : pd.DataFrame
        Feature dataframe
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
        - binary_predictions: 0/1 predictions
        - probabilities: probability of positive class
    """
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        # Works for sklearn models and CatBoost
        probabilities = model.predict_proba(X)
        if len(probabilities.shape) > 1:
            probabilities = probabilities[:, 1]
    elif hasattr(model, 'predict'):
        # For LightGBM and models that directly return probabilities
        probabilities = model.predict(X, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
        if len(probabilities.shape) > 1:
            probabilities = probabilities[:, 1]
    else:
        raise ValueError("Model must have either predict_proba or predict method")
    
    # Convert to binary predictions
    binary_predictions = (probabilities > threshold).astype(int)
    
    return binary_predictions, probabilities

def predict_lightgbm(model: lgb.Booster, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with LightGBM model.
    
    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    X : pd.DataFrame
        Feature dataframe
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
    """
    # LightGBM returns probabilities directly
    probabilities = model.predict(X, num_iteration=model.best_iteration)
    binary_predictions = (probabilities > threshold).astype(int)
    
    return binary_predictions, probabilities

def predict_catboost(model: cb.CatBoostClassifier, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with CatBoost model.
    
    Parameters:
    -----------
    model : cb.CatBoostClassifier
        Trained CatBoost model
    X : pd.DataFrame
        Feature dataframe
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
    """
    probabilities = model.predict_proba(X)[:, 1]
    binary_predictions = (probabilities > threshold).astype(int)
    
    return binary_predictions, probabilities

def predict_sklearn(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with sklearn models (RandomForest, etc.).
    
    Parameters:
    -----------
    model : sklearn model
        Trained sklearn model with predict_proba method
    X : pd.DataFrame
        Feature dataframe
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
    """
    probabilities = model.predict_proba(X)[:, 1]
    binary_predictions = (probabilities > threshold).astype(int)
    
    return binary_predictions, probabilities

def load_and_predict(model_path: str, X: pd.DataFrame, model_type: str = 'auto', threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a saved model and make predictions.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model file
    X : pd.DataFrame
        Feature dataframe
    model_type : str
        Type of model ('lightgbm', 'catboost', 'sklearn', 'auto')
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
    """
    model_path = Path(model_path)
    
    # Auto-detect model type based on file extension
    if model_type == 'auto':
        if model_path.suffix in ['.txt', '.lgb']:
            model_type = 'lightgbm'
        elif model_path.suffix in ['.cbm', '.catboost']:
            model_type = 'catboost'
        elif model_path.suffix in ['.pkl', '.joblib']:
            model_type = 'sklearn'
        else:
            raise ValueError(f"Cannot auto-detect model type for extension {model_path.suffix}")
    
    # Load model based on type
    if model_type == 'lightgbm':
        model = lgb.Booster(model_file=str(model_path))
        return predict_lightgbm(model, X, threshold)
    elif model_type == 'catboost':
        model = cb.CatBoostClassifier()
        model.load_model(str(model_path))
        return predict_catboost(model, X, threshold)
    elif model_type == 'sklearn':
        model = joblib.load(model_path)
        return predict_sklearn(model, X, threshold)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_submission(
    predictions: Union[np.ndarray, pd.Series],
    test_ids: pd.Series,
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """
    Create submission file for competition.
    
    Parameters:
    -----------
    predictions : array-like
        Predictions (either binary or probabilities)
    test_ids : pd.Series
        Test set IDs
    output_path : str
        Path to save submission file
        
    Returns:
    --------
    pd.DataFrame : Submission dataframe
    """
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
    method: str = 'proba_averaging',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ensemble predictions from multiple models.
    
    Parameters:
    -----------
    models : list
        List of trained models
    X : pd.DataFrame
        Feature dataframe
    weights : list
        Weights for each model (default: equal weights)
    method : str
        Ensemble method ('voting' or 'proba_averaging')
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    tuple : (binary_predictions, probabilities)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Collect predictions from all models
    all_probas = []
    for model in models:
        _, probas = predict_with_model(model, X, threshold)
        all_probas.append(probas)
    
    # Stack predictions
    probas_array = np.column_stack(all_probas)
    
    if method == 'voting':
        # Hard voting: majority vote
        binary_preds = (probas_array > threshold).astype(int)
        ensemble_pred = np.round(np.average(binary_preds, axis=1, weights=weights))
        ensemble_proba = np.average(probas_array, axis=1, weights=weights)
    
    elif method == 'proba_averaging':
        # Soft voting: average probabilities
        ensemble_proba = np.average(probas_array, axis=1, weights=weights)
        ensemble_pred = (ensemble_proba > threshold).astype(int)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred.astype(int), ensemble_proba

def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame = None,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with predictions, metrics, and feature importance.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    X_train : pd.DataFrame, optional
        Training features (for feature names in importance)
    model_name : str
        Name of the model for display
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    dict : Dictionary containing:
        - predictions: (binary, probabilities)
        - metrics: dict of performance metrics
        - feature_importance: DataFrame of top features
        - confusion_matrix: numpy array
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if verbose:
        print("\n" + "="*60)
        print(f"{model_name.upper()} EVALUATION")
        print("="*60)
        print("\nGenerating predictions...")
    
    # Make predictions
    y_pred_binary, y_pred_proba = predict_with_model(model, X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary, zero_division=0),
        'recall': recall_score(y_test, y_pred_binary, zero_division=0),
        'f1': f1_score(y_test, y_pred_binary, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    if verbose:
        print("\nModel Performance Metrics:")
        print("-"*40)
        for metric, value in metrics.items():
            print(f"{metric:12s}: {value:.4f}")
        print(f"\n✓ Final Test AUC: {metrics['roc_auc']:.4f}")
    
    # Get feature importance if possible
    feature_importance_df = None
    if X_train is not None and hasattr(model, 'feature_importance'):
        try:
            # For LightGBM
            importance_gain = model.feature_importance(importance_type="gain")
            feature_importance_df = pd.DataFrame({
                "feature": X_train.columns,
                "importance": importance_gain
            }).sort_values("importance", ascending=False).head(20)
            feature_importance_df["importance_normalized"] = (
                feature_importance_df["importance"] / feature_importance_df["importance"].sum() * 100
            )
            
            if verbose:
                print("\nTop 20 Feature Importance:")
                print("="*50)
                for _, row in feature_importance_df.iterrows():
                    print(f"{row['feature']:30s} {row['importance']:8.0f} ({row['importance_normalized']:5.1f}%)")
        except:
            pass
    elif hasattr(model, 'feature_importances_'):
        try:
            # For sklearn models
            feature_importance_df = pd.DataFrame({
                "feature": X_train.columns if X_train is not None else [f"f{i}" for i in range(len(model.feature_importances_))],
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False).head(20)
            feature_importance_df["importance_normalized"] = (
                feature_importance_df["importance"] / feature_importance_df["importance"].sum() * 100
            )
            
            if verbose:
                print("\nTop 20 Feature Importance:")
                print("="*50)
                for _, row in feature_importance_df.iterrows():
                    print(f"{row['feature']:30s} {row['importance']:8.4f} ({row['importance_normalized']:5.1f}%)")
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    if verbose:
        # Classification report
        print("\n\nClassification Report:")
        print("="*50)
        print(classification_report(y_test, y_pred_binary, 
                                  target_names=['Not Drafted', 'Drafted'],
                                  zero_division=0))
        
        print("\nConfusion Matrix:")
        print("="*50)
        print(f"True Negatives:  {cm[0,0]:5d}")
        print(f"False Positives: {cm[0,1]:5d}")
        print(f"False Negatives: {cm[1,0]:5d}")
        print(f"True Positives:  {cm[1,1]:5d}")
    
    return {
        'predictions': (y_pred_binary, y_pred_proba),
        'metrics': metrics,
        'feature_importance': feature_importance_df,
        'confusion_matrix': cm
    }

def business_impact_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list = [0.3, 0.5, 0.7, 0.9],
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze business impact of model at different decision thresholds.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    thresholds : list
        Probability thresholds to analyze
    model_name : str
        Name of the model for display
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Dictionary containing threshold analysis and recommendations
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    if verbose:
        print("="*60)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*60)
    
    # Get predictions
    _, y_test_proba = predict_with_model(model, X_test)
    
    # Calculate basic statistics
    test_size = len(y_test)
    drafted_players = int(y_test.sum())
    draft_rate = drafted_players / test_size
    
    if verbose:
        print(f"\nTest Set Statistics:")
        print(f"  Total players evaluated: {test_size}")
        print(f"  Actually drafted: {drafted_players} ({draft_rate:.2%})")
        print(f"  Draft rate: {draft_rate:.2%}")
    
    # Analyze at different thresholds
    threshold_analysis = []
    
    if verbose:
        print(f"\nPrediction Analysis at Different Confidence Levels:")
        print("-"*60)
        print(f"{'Threshold':<12} {'Predicted':<10} {'Correct':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-"*60)
    
    for threshold in thresholds:
        predictions = (y_test_proba > threshold).astype(int)
        correct = ((predictions == 1) & (y_test == 1)).sum()
        total_predicted = predictions.sum()
        
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        threshold_analysis.append({
            'threshold': threshold,
            'predicted': total_predicted,
            'correct': correct,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if verbose:
            print(f"{threshold:<12.1f} {total_predicted:<10} {correct:<10} {precision:<12.2%} {recall:<10.2%} {f1:<10.3f}")
    
    # Business scenario analysis
    if verbose:
        print(f"\n" + "="*60)
        print("BUSINESS SCENARIO ANALYSIS")
        print("="*60)
        
        # Conservative approach (high threshold)
        high_threshold = 0.7
        predictions_high = (y_test_proba > high_threshold).astype(int)
        tp_high = ((predictions_high == 1) & (y_test == 1)).sum()
        fp_high = ((predictions_high == 1) & (y_test == 0)).sum()
        fn_high = ((predictions_high == 0) & (y_test == 1)).sum()
        
        print(f"\n1. Conservative Approach (threshold={high_threshold}):")
        print(f"   - Focus on high-confidence predictions only")
        print(f"   - Would scout {tp_high + fp_high} players intensively")
        print(f"   - Correctly identifies {tp_high}/{drafted_players} drafted players ({tp_high/drafted_players:.1%})")
        print(f"   - False positives: {fp_high} (wasted effort on {fp_high/(tp_high+fp_high):.1%} of scouted players)" if (tp_high+fp_high) > 0 else "")
        print(f"   - Misses {fn_high} future draft picks")
        
        # Balanced approach (medium threshold)
        med_threshold = 0.5
        predictions_med = (y_test_proba > med_threshold).astype(int)
        tp_med = ((predictions_med == 1) & (y_test == 1)).sum()
        fp_med = ((predictions_med == 1) & (y_test == 0)).sum()
        fn_med = ((predictions_med == 0) & (y_test == 1)).sum()
        
        print(f"\n2. Balanced Approach (threshold={med_threshold}):")
        print(f"   - Balance between precision and recall")
        print(f"   - Would scout {tp_med + fp_med} players intensively")
        print(f"   - Correctly identifies {tp_med}/{drafted_players} drafted players ({tp_med/drafted_players:.1%})")
        print(f"   - False positives: {fp_med} (wasted effort on {fp_med/(tp_med+fp_med):.1%} of scouted players)" if (tp_med+fp_med) > 0 else "")
        print(f"   - Misses {fn_med} future draft picks")
        
        # Aggressive approach (low threshold)
        low_threshold = 0.3
        predictions_low = (y_test_proba > low_threshold).astype(int)
        tp_low = ((predictions_low == 1) & (y_test == 1)).sum()
        fp_low = ((predictions_low == 1) & (y_test == 0)).sum()
        fn_low = ((predictions_low == 0) & (y_test == 1)).sum()
        
        print(f"\n3. Aggressive Approach (threshold={low_threshold}):")
        print(f"   - Minimize missed opportunities")
        print(f"   - Would scout {tp_low + fp_low} players intensively")
        print(f"   - Correctly identifies {tp_low}/{drafted_players} drafted players ({tp_low/drafted_players:.1%})")
        print(f"   - False positives: {fp_low} (wasted effort on {fp_low/(tp_low+fp_low):.1%} of scouted players)" if (tp_low+fp_low) > 0 else "")
        print(f"   - Misses {fn_low} future draft picks")
        
        # Recommendations
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find optimal threshold for different objectives
        best_precision_idx = max(range(len(threshold_analysis)), 
                                 key=lambda i: threshold_analysis[i]['precision'])
        best_recall_idx = max(range(len(threshold_analysis)), 
                             key=lambda i: threshold_analysis[i]['recall'])
        best_f1_idx = max(range(len(threshold_analysis)), 
                         key=lambda i: threshold_analysis[i]['f1'])
        
        print(f"\n• For maximum precision (minimize false positives):")
        print(f"  Use threshold = {threshold_analysis[best_precision_idx]['threshold']:.1f}")
        print(f"  Achieves {threshold_analysis[best_precision_idx]['precision']:.1%} precision")
        
        print(f"\n• For maximum recall (minimize missed talent):")
        print(f"  Use threshold = {threshold_analysis[best_recall_idx]['threshold']:.1f}")
        print(f"  Achieves {threshold_analysis[best_recall_idx]['recall']:.1%} recall")
        
        print(f"\n• For best balance (F1 score):")
        print(f"  Use threshold = {threshold_analysis[best_f1_idx]['threshold']:.1f}")
        print(f"  Achieves F1 = {threshold_analysis[best_f1_idx]['f1']:.3f}")
        
        print(f"\nKey Insights:")
        print(f"• The model can effectively identify draft prospects")
        print(f"• Threshold selection depends on organizational priorities:")
        print(f"  - Limited resources → Use higher threshold")
        print(f"  - Comprehensive coverage → Use lower threshold")
        print(f"• Consider using probability scores for tiered scouting intensity")
    
    return {
        'test_statistics': {
            'total_players': test_size,
            'drafted_players': drafted_players,
            'draft_rate': draft_rate
        },
        'threshold_analysis': threshold_analysis,
        'probabilities': y_test_proba
    }

# Backward compatibility
def make_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """Make binary predictions using trained model (backward compatibility)."""
    binary_pred, _ = predict_with_model(model, X)
    return binary_pred

def make_predictions_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Make probability predictions using trained model (backward compatibility)."""
    _, probas = predict_with_model(model, X)
    return probas