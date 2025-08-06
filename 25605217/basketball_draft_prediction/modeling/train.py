"""Model training utilities with hyperparameter optimization."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import optuna
from optuna.integration import LightGBMPruningCallback, CatBoostPruningCallback
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_lightgbm_base_params(random_state: int = 42, is_unbalanced: bool = True) -> Dict[str, Any]:
    """
    Get base parameters for LightGBM model.
    
    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility
    is_unbalanced : bool
        Whether to handle class imbalance
        
    Returns:
    --------
    dict : Base parameters for LightGBM
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_threads': -1,
        'verbosity': -1,
        'is_unbalanced': is_unbalanced,
        'seed': random_state
    }
    
    print("Base LightGBM parameters set:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\nOptuna will optimize additional hyperparameters including:")
    print("  - num_leaves, max_depth")
    print("  - learning_rate, feature_fraction")  
    print("  - bagging_fraction, bagging_freq")
    print("  - min_child_samples, regularization parameters")
    
    return params

def prepare_lgb_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[lgb.Dataset, lgb.Dataset, List[int]]:
    """
    Prepare LightGBM datasets with proper categorical feature handling.
    
    Parameters:
    -----------
    X_train, X_val : pd.DataFrame
        Feature dataframes
    y_train, y_val : pd.Series
        Target variables
        
    Returns:
    --------
    tuple : (lgb_train, lgb_val, cat_feature_indices)
    """
    # Get indices of categorical features
    cat_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype.name == 'category':
            cat_feature_indices.append(i)
    
    print(f"Categorical feature indices: {cat_feature_indices}")
    if cat_feature_indices:
        print(f"Categorical columns: {[X_train.columns[i] for i in cat_feature_indices]}")
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(
        X_train, 
        label=y_train, 
        categorical_feature=cat_feature_indices if cat_feature_indices else 'auto'
    )
    lgb_val = lgb.Dataset(
        X_val, 
        label=y_val, 
        categorical_feature=cat_feature_indices if cat_feature_indices else 'auto',
        reference=lgb_train
    )
    
    return lgb_train, lgb_val, cat_feature_indices

def train_lightgbm_basic(
    lgb_train: lgb.Dataset,
    lgb_val: lgb.Dataset,
    params: Dict[str, Any],
    num_boost_round: int = 100,
    early_stopping_rounds: int = 30,
    verbose_eval: int = 20
) -> lgb.Booster:
    """
    Train basic LightGBM model with early stopping.
    
    Parameters:
    -----------
    lgb_train : lgb.Dataset
        Training dataset
    lgb_val : lgb.Dataset
        Validation dataset
    params : dict
        Model parameters
    num_boost_round : int
        Maximum number of boosting rounds
    early_stopping_rounds : int
        Early stopping patience
    verbose_eval : int
        Print metrics every N rounds
        
    Returns:
    --------
    lgb.Booster : Trained model
    """
    print("\nStarting LightGBM training with base parameters...")
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose_eval)
        ]
    )
    
    print("\n✓ Model training completed successfully")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score['valid_0'][params.get('metric', 'auc')]:.4f}")
    
    return model

def train_lightgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42,
    base_params: Optional[Dict[str, Any]] = None
) -> Tuple[lgb.Booster, Dict[str, Any], Dict[str, float]]:
    """
    Train LightGBM with Optuna hyperparameter optimization.
    
    Parameters:
    -----------
    X_train, X_val : pd.DataFrame
        Feature dataframes
    y_train, y_val : pd.Series
        Target variables
    n_trials : int
        Number of Optuna trials
    random_state : int
        Random seed
    base_params : dict
        Base parameters for LightGBM
        
    Returns:
    --------
    tuple : (model, best_params, metrics)
    """
    import optuna.integration.lightgbm as lgb_optuna
    
    # Prepare datasets
    lgb_train, lgb_val, cat_indices = prepare_lgb_datasets(
        X_train, y_train, X_val, y_val
    )
    
    # Get base parameters if not provided
    if base_params is None:
        base_params = get_lightgbm_base_params(random_state)
    
    print("\nStarting LightGBM Optuna optimization...")
    print(f"Running {n_trials} trials to find optimal hyperparameters...")
    
    # Use Optuna's LightGBMTunerCV for optimization
    tuner = lgb_optuna.LightGBMTunerCV(
        params=base_params,
        train_set=lgb_train,
        num_boost_round=500,
        folds=None,  # Use validation set instead of CV
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        optuna_seed=random_state,
        show_progress_bar=False
    )
    
    # Run optimization
    tuner.run()
    
    # Get best parameters
    best_params = tuner.best_params
    best_params.update(base_params)
    
    print("\n✓ Hyperparameter optimization completed")
    print(f"Best score: {tuner.best_score:.4f}")
    print("\nOptimized parameters:")
    for key, value in best_params.items():
        if key not in base_params:
            print(f"  {key}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = lgb.train(
        best_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Calculate metrics
    y_pred_proba = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    print("\nValidation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return final_model, best_params, metrics

def train_catboost_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42
) -> Tuple[cb.CatBoostClassifier, Dict[str, Any], Dict[str, float]]:
    """Train CatBoost with Optuna hyperparameter optimization."""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'random_seed': random_state,
            'verbose': False,
            'thread_count': -1
        }
        
        pruning_callback = CatBoostPruningCallback(trial, 'Logloss')
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            callbacks=[pruning_callback],
            early_stopping_rounds=50,
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['random_seed'] = random_state
    best_params['verbose'] = False
    best_params['thread_count'] = -1
    
    best_model = cb.CatBoostClassifier(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    metrics = evaluate_model(best_model, X_val, y_val)
    
    return best_model, best_params, metrics

def train_random_forest_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any], Dict[str, float]]:
    """Train Random Forest with Optuna hyperparameter optimization."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': random_state,
            'n_jobs': -1
        }
        
        if params['bootstrap']:
            params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['n_jobs'] = -1
    
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    metrics = evaluate_model(best_model, X_val, y_val)
    
    return best_model, best_params, metrics

def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba)
    }
    
    return metrics

def save_model(model, model_name: str, save_path: str = "models"):
    """Save trained model to disk."""
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path: str):
    """Load model from disk."""
    return joblib.load(model_path)


class LightGBMTrainer:
    """
    Comprehensive LightGBM trainer with multiple training methods.
    
    This class provides various training approaches:
    - Basic training without optimization
    - Standard Optuna optimization
    - Optuna integration with pruning callback
    - LightGBMTunerCV for automatic hyperparameter tuning
    """
    
    def __init__(self, random_state=42, is_unbalanced=True, verbose=False):
        """
        Initialize the LightGBM trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        is_unbalanced : bool
            Whether to handle class imbalance
        verbose : bool
            Whether to print detailed logs
        """
        self.random_state = random_state
        self.is_unbalanced = is_unbalanced
        self.verbose = verbose
        self.model = None
        self.best_params = None
        self.metrics = None
        
    def train_basic(self, X_train, y_train, X_val, y_val, 
                   params=None, num_boost_round=100, 
                   early_stopping_rounds=30, verbose_eval=20):
        """
        Train basic LightGBM model without optimization.
        
        Parameters:
        -----------
        X_train, X_val : pd.DataFrame
            Feature dataframes
        y_train, y_val : pd.Series
            Target variables
        params : dict
            Model parameters (uses defaults if None)
        num_boost_round : int
            Maximum number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        verbose_eval : int
            Print metrics every N rounds
            
        Returns:
        --------
        lgb.Booster : Trained model
        """
        # Prepare datasets
        lgb_train, lgb_val, cat_indices = prepare_lgb_datasets(
            X_train, y_train, X_val, y_val
        )
        
        # Get parameters
        if params is None:
            params = get_lightgbm_base_params(self.random_state, self.is_unbalanced)
        
        # Train model
        self.model = train_lightgbm_basic(
            lgb_train, lgb_val, params,
            num_boost_round, early_stopping_rounds, 
            verbose_eval if self.verbose else 0
        )
        
        # Calculate metrics
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def train_with_optuna(self, X_train, y_train, X_val, y_val, 
                         n_trials=100, use_pruning=False):
        """
        Train LightGBM with standard Optuna optimization.
        
        Parameters:
        -----------
        X_train, X_val : pd.DataFrame
            Feature dataframes
        y_train, y_val : pd.Series
            Target variables
        n_trials : int
            Number of Optuna trials
        use_pruning : bool
            Whether to use pruning for early stopping of bad trials
            
        Returns:
        --------
        lgb.Booster : Trained model with optimized hyperparameters
        """
        # Prepare datasets
        lgb_train, lgb_val, cat_indices = prepare_lgb_datasets(
            X_train, y_train, X_val, y_val
        )
        
        def objective(trial):
            params = get_lightgbm_base_params(self.random_state, self.is_unbalanced)
            
            # Suggest hyperparameters
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            })
            
            # Train with pruning callback if requested
            callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
            if use_pruning:
                callbacks.append(LightGBMPruningCallback(trial, 'auc'))
            
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=500,
                callbacks=callbacks
            )
            
            # Return AUC score
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            return roc_auc_score(y_val, y_pred_proba)
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        if self.verbose:
            print(f"\nStarting Optuna optimization with {n_trials} trials...")
        
        study.optimize(
            objective, 
            n_trials=n_trials, 
            show_progress_bar=self.verbose
        )
        
        # Get best parameters and train final model
        self.best_params = study.best_params
        final_params = get_lightgbm_base_params(self.random_state, self.is_unbalanced)
        final_params.update(self.best_params)
        
        if self.verbose:
            print(f"\nBest trial score: {study.best_value:.4f}")
            print("\nBest parameters:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")
        
        # Train final model with best parameters
        self.model = lgb.train(
            final_params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50 if self.verbose else 0)]
        )
        
        # Calculate metrics
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def train_with_optuna_integration(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Train LightGBM using Optuna integration with LightGBMPruningCallback.
        
        Parameters:
        -----------
        X_train, X_val : pd.DataFrame
            Feature dataframes
        y_train, y_val : pd.Series
            Target variables
        n_trials : int
            Number of Optuna trials
            
        Returns:
        --------
        lgb.Booster : Trained model with optimized hyperparameters
        """
        # This method uses pruning by default
        return self.train_with_optuna(X_train, y_train, X_val, y_val, 
                                     n_trials=n_trials, use_pruning=True)
    
    def train_with_tuner_cv(self, X_train, y_train, X_val, y_val, 
                           time_budget=3600, n_folds=5):
        """
        Train LightGBM using LightGBMTunerCV for automatic hyperparameter tuning.
        
        Parameters:
        -----------
        X_train, X_val : pd.DataFrame
            Feature dataframes
        y_train, y_val : pd.Series
            Target variables
        time_budget : int
            Time budget in seconds for optimization
        n_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        lgb.Booster : Trained model with optimized hyperparameters
        """
        import optuna.integration.lightgbm as lgb_optuna
        
        # Prepare datasets
        lgb_train, lgb_val, cat_indices = prepare_lgb_datasets(
            X_train, y_train, X_val, y_val
        )
        
        # Get base parameters
        base_params = get_lightgbm_base_params(self.random_state, self.is_unbalanced)
        
        if self.verbose:
            print(f"\nStarting LightGBMTunerCV optimization...")
            print(f"Time budget: {time_budget} seconds")
            print(f"Cross-validation folds: {n_folds}")
        
        # Use TunerCV for optimization
        tuner = lgb_optuna.LightGBMTunerCV(
            params=base_params,
            train_set=lgb_train,
            num_boost_round=500,
            folds=n_folds,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            time_budget=time_budget,
            optuna_seed=self.random_state,
            show_progress_bar=self.verbose
        )
        
        # Run optimization
        tuner.run()
        
        # Get best parameters
        self.best_params = tuner.best_params
        self.best_params.update(base_params)
        
        if self.verbose:
            print(f"\nBest score: {tuner.best_score:.4f}")
            print("\nOptimized parameters:")
            for key, value in tuner.best_params.items():
                if key not in base_params:
                    print(f"  {key}: {value}")
        
        # Train final model with best parameters
        self.model = lgb.train(
            self.best_params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50 if self.verbose else 0)]
        )
        
        # Calculate metrics
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Normalize importance
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
            
        Returns:
        --------
        tuple : (predicted_classes, predicted_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_pred, y_pred_proba
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        self.model.save_model(filepath)
        
        # Also save parameters and metrics
        import json
        metadata = {
            'best_params': self.best_params,
            'metrics': self.metrics,
            'random_state': self.random_state,
            'is_unbalanced': self.is_unbalanced
        }
        
        metadata_path = filepath.replace('.txt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
            print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = lgb.Booster(model_file=filepath)
        
        # Load metadata if available
        import json
        metadata_path = filepath.replace('.txt', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_params = metadata.get('best_params')
                self.metrics = metadata.get('metrics')
                self.random_state = metadata.get('random_state', self.random_state)
                self.is_unbalanced = metadata.get('is_unbalanced', self.is_unbalanced)
        
        if self.verbose:
            print(f"Model loaded from {filepath}")


class CatBoostTrainer:
    """
    CatBoost trainer with Optuna integration for NBA draft prediction.
    """
    
    def __init__(self, random_state=42, verbose=False):
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.best_params = None
        self.metrics = None
        
    def train_basic(self, X_train, y_train, X_val, y_val, 
                   iterations=1000, early_stopping_rounds=50):
        """
        Train basic CatBoost model without optimization.
        """
        from catboost import CatBoostClassifier, Pool
        
        # Identify categorical features
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create Pool objects
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Base parameters
        params = {
            'iterations': iterations,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': self.random_state,
            'early_stopping_rounds': early_stopping_rounds,
            'use_best_model': True,
            'verbose': self.verbose,
            'auto_class_weights': 'Balanced'  # Handle class imbalance
        }
        
        # Train model
        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=val_pool, verbose=self.verbose)
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def train_with_optuna(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Train CatBoost with Optuna optimization.
        """
        from catboost import CatBoostClassifier, Pool
        
        # Identify categorical features
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': self.random_state,
                'early_stopping_rounds': 50,
                'use_best_model': True,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }
            
            # Create Pool objects
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            
            # Train model
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            
            # Return AUC score
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        if self.verbose:
            print(f"\nStarting Optuna optimization with {n_trials} trials...")
        
        study.optimize(
            objective, 
            n_trials=n_trials, 
            show_progress_bar=self.verbose
        )
        
        # Get best parameters and train final model
        self.best_params = study.best_params
        
        if self.verbose:
            print(f"\nBest trial score: {study.best_value:.4f}")
            print("\nBest parameters:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")
        
        # Train final model with best parameters
        final_params = {
            'iterations': 1000,
            **self.best_params,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': self.random_state,
            'early_stopping_rounds': 50,
            'use_best_model': True,
            'verbose': self.verbose,
            'auto_class_weights': 'Balanced'
        }
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        self.model = CatBoostClassifier(**final_params)
        self.model.fit(train_pool, eval_set=val_pool, verbose=self.verbose)
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from the trained model.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Normalize importance
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        
        return y_pred, y_pred_proba
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        self.model.save_model(filepath)
        
        # Save metadata
        import json
        metadata = {
            'best_params': self.best_params,
            'metrics': self.metrics,
            'random_state': self.random_state
        }
        
        metadata_path = str(filepath).replace('.cbm', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
            print(f"Metadata saved to {metadata_path}")


class RandomForestTrainer:
    """
    Random Forest trainer with Optuna optimization for NBA draft prediction.
    """
    
    def __init__(self, random_state=42, verbose=False):
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.best_params = None
        self.metrics = None
        
    def train_basic(self, X_train, y_train, X_val, y_val, 
                   n_estimators=100, max_depth=None):
        """
        Train basic Random Forest model without optimization.
        """
        # Handle categorical features by encoding
        X_train_encoded = self._encode_categoricals(X_train)
        X_val_encoded = self._encode_categoricals(X_val)
        
        # Base parameters
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'class_weight': 'balanced',  # Handle class imbalance
            'n_jobs': -1
        }
        
        # Train model
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train_encoded, y_train)
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_val_encoded)[:, 1]
        y_pred = self.model.predict(X_val_encoded)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def train_with_optuna(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Train Random Forest with Optuna optimization.
        """
        # Handle categorical features
        X_train_encoded = self._encode_categoricals(X_train)
        X_val_encoded = self._encode_categoricals(X_val)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train_encoded, y_train)
            
            # Return AUC score
            y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        if self.verbose:
            print(f"\nStarting Optuna optimization with {n_trials} trials...")
        
        study.optimize(
            objective, 
            n_trials=n_trials, 
            show_progress_bar=self.verbose
        )
        
        # Get best parameters and train final model
        self.best_params = study.best_params
        
        if self.verbose:
            print(f"\nBest trial score: {study.best_value:.4f}")
            print("\nBest parameters:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")
        
        # Train final model with best parameters
        final_params = {
            **self.best_params,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        
        self.model = RandomForestClassifier(**final_params)
        self.model.fit(X_train_encoded, y_train)
        
        # Store encoded data for predictions
        self._X_train_encoded = X_train_encoded
        self._X_val_encoded = X_val_encoded
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_val_encoded)[:, 1]
        y_pred = self.model.predict(X_val_encoded)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        if self.verbose:
            print("\nValidation metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return self.model
    
    def _encode_categoricals(self, X):
        """
        Encode categorical features for Random Forest.
        """
        X_encoded = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        
        return X_encoded
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from the trained model.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Normalize importance
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        X_encoded = self._encode_categoricals(X)
        y_pred_proba = self.model.predict_proba(X_encoded)[:, 1]
        y_pred = self.model.predict(X_encoded)
        
        return y_pred, y_pred_proba
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        joblib.dump(self.model, filepath)
        
        # Save metadata
        import json
        metadata = {
            'best_params': self.best_params,
            'metrics': self.metrics,
            'random_state': self.random_state
        }
        
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
            print(f"Metadata saved to {metadata_path}")