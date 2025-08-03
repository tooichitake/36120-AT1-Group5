import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze data"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Convert all numeric columns to float64 for competition precision
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        train_df[col] = train_df[col].astype(np.float64)
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(np.float64)
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print(f"\nTarget distribution:")
    drafted_count = train_df['drafted'].sum()
    total_count = len(train_df)
    print(f"Drafted: {int(drafted_count)} ({drafted_count/total_count:.3%})")
    print(f"Not drafted: {total_count - int(drafted_count)} ({(total_count - drafted_count)/total_count:.3%})")
    
    return train_df, test_df

def height_to_inches(height_str):
    """Convert height string to inches"""
    if pd.isna(height_str) or height_str == '-':
        return np.nan  # Keep as NaN for LightGBM to handle
    if isinstance(height_str, (int, float)):
        return height_str
    
    # Handle "6-Jun" format (6'6")
    parts = str(height_str).split('-')
    if len(parts) == 2:
        try:
            feet = int(parts[0])
            month_to_inches = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            inches = month_to_inches.get(parts[1], 0)
            return feet * 12 + inches
        except:
            return np.nan
    return np.nan

def create_features_for_lightgbm(df):
    """Create features for LightGBM, preserving missing values"""
    df_feat = df.copy()
    
    print("\n=== LightGBM Native Feature Engineering ===")
    
    # 1. Height processing - do not fill missing values
    df_feat['height_numeric'] = df_feat['ht'].apply(height_to_inches).astype(np.float64, errors='ignore')
    
    # 2. Basic interaction features
    df_feat['usage_efficiency'] = (df_feat['usg'] * df_feat['TS_per']).astype(np.float64)
    df_feat['minutes_impact'] = (df_feat['Min_per'] * df_feat['bpm']).astype(np.float64)
    df_feat['offensive_load'] = (df_feat['usg'] + df_feat['AST_per']).astype(np.float64)
    
    # 3. Ratio features
    df_feat['assist_turnover_ratio'] = (df_feat['AST_per'] / (df_feat['TO_per'] + 0.1)).astype(np.float64)
    df_feat['rebound_total'] = (df_feat['ORB_per'] + df_feat['DRB_per']).astype(np.float64)
    df_feat['defensive_stats'] = (df_feat['stl_per'] + df_feat['blk_per']).astype(np.float64)
    
    # 4. Advanced features
    df_feat['all_around_score'] = (
        df_feat['AST_per'] * 0.3 + 
        df_feat['rebound_total'] * 0.4 + 
        df_feat['defensive_stats'] * 0.3
    ).astype(np.float64)
    
    # 5. Efficiency metrics
    df_feat['true_shooting_volume'] = (df_feat['TS_per'] * np.log1p(df_feat['usg'])).astype(np.float64)
    df_feat['per_minute_impact'] = (df_feat['bpm'] / (df_feat['Min_per'] + 1)).astype(np.float64)
    
    # 6. Grade to numeric conversion
    yr_to_numeric = {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}
    df_feat['yr_numeric'] = df_feat['yr'].map(yr_to_numeric)
    
    # 7. Conference strength
    power_conferences = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']
    df_feat['power_conference'] = df_feat['conf'].isin(power_conferences).astype(int)
    
    # 8. Rare skill indicators
    df_feat['big_man_shooter'] = ((df_feat['height_numeric'] > 80) & (df_feat['TP_per'] > 35)).astype(int)
    df_feat['playmaking_big'] = ((df_feat['height_numeric'] > 80) & (df_feat['AST_per'] > 20)).astype(int)
    
    # 9. Preserve original missing value indicators
    df_feat['has_recruit_rank'] = df_feat['Rec_Rank'].notna().astype(int)
    
    print(f"Feature engineering completed, new features added: {len([col for col in df_feat.columns if col not in df.columns])}")
    print(f"Columns with missing values: {df_feat.isnull().any().sum()}")
    
    return df_feat

def prepare_lightgbm_datasets(train_feat, test_feat):
    """Prepare LightGBM native datasets"""
    
    # Select features - exclude ID and target variables
    exclude_cols = ['player_id', 'drafted', 'ht']
    
    # Categorical features
    categorical_features = ['team', 'conf', 'yr', 'type']
    
    # Handle categorical features - LightGBM supports direct category type
    for cat_feat in categorical_features:
        train_feat[cat_feat] = train_feat[cat_feat].astype('category')
        test_feat[cat_feat] = test_feat[cat_feat].astype('category')
        
        # Ensure train and test sets have same categories
        all_cats = set(train_feat[cat_feat].cat.categories) | set(test_feat[cat_feat].cat.categories)
        train_feat[cat_feat] = train_feat[cat_feat].cat.set_categories(all_cats)
        test_feat[cat_feat] = test_feat[cat_feat].cat.set_categories(all_cats)
    
    # All features
    feature_cols = [col for col in train_feat.columns if col not in exclude_cols]
    
    X_train = train_feat[feature_cols]
    y_train = train_feat['drafted']
    X_test = test_feat[feature_cols]
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    print(f"Categorical features: {categorical_features}")
    print(f"Training set shape: {X_train.shape}")
    
    return X_train, y_train, X_test, feature_cols, categorical_features

def main():
    """Main function"""
    print("=== NBA Draft Prediction - LightGBM Optuna Native Integration ===\n")
    
    # 1. Load data
    train_df, test_df = load_and_analyze_data()
    
    # 2. Feature engineering - preserve missing values
    print("\n=== Feature Engineering ===")
    train_feat = create_features_for_lightgbm(train_df)
    test_feat = create_features_for_lightgbm(test_df)
    
    # 3. Prepare datasets
    X_train, y_train, X_test, feature_cols, categorical_features = prepare_lightgbm_datasets(train_feat, test_feat)
    
    # 4. Set basic parameters (only necessary fixed parameters)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'dart',  # Most stable for tabular data, can try gbdt, dart
        'num_threads': -1,
        'verbosity': -1,
        'is_unbalanced': True,  # Handle class imbalance
        'seed': 42,  # Random seed
    }
    
    # 5. Create dataset - ensure float64 precision
    # Convert to float64 before creating dataset
    numeric_features = [col for col in X_train.columns if col not in categorical_features]
    for col in numeric_features:
        X_train[col] = X_train[col].astype(np.float64)
        X_test[col] = X_test[col].astype(np.float64)
    
    lgb_train = lgb.Dataset(X_train, label=y_train.astype(np.float64), categorical_feature=categorical_features)
    
    # 6. Use Optuna's LightGBM integration for hyperparameter optimization
    print("\n=== LightGBM Optuna Native Optimization ===")
    print("Optuna will automatically search for best hyperparameters...")
    
    # Use optuna.integration.lightgbm's LightGBMTunerCV for optimization
    tuner = lgb_optuna.LightGBMTunerCV(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=500,
        nfold=10,  # 5-fold cross-validation
        stratified=True,  # Stratified sampling (for classification)
        shuffle=True,  # Shuffle data
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(0)  # Don't show training logs
        ],
        show_progress_bar=False,  # Don't show progress bar for clean output
        return_cvbooster=True  # Allow getting best model
    )
    
    # Run optimization
    print("Starting hyperparameter optimization (this may take several minutes)...")
    print("Optimization progress:")
    tuner.run()
    print("✓ Hyperparameter optimization completed")
    
    # Get best parameters
    best_params = tuner.best_params
    best_params.update(lgb_params)
    
    print("\n=== Best Parameters ===")
    for key, value in best_params.items():
        if key not in lgb_params:  # Only show optimized parameters
            print(f"  {key}: {value}")
    
    # Get best model (CVBooster)
    print("\n=== Getting Best Cross-Validation Model ===")
    try:
        cv_booster = tuner.get_best_booster()
        print("✓ Successfully obtained best CVBooster")
        # Use first model from CVBooster for prediction
        model = cv_booster.boosters[0]
    except ValueError:
        print("Unable to get CVBooster, retraining with best parameters...")
        model = lgb.train(
            best_params,
            lgb_train,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(0)]
        )
    
    # 7. Show best score
    print(f"\nBest cross-validation AUC: {tuner.best_score:.4f}")
    
    # 8. Feature importance
    print("\n=== Top 20 Feature Importance ===")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    
    print(importance_df.to_string(index=False))
    
    # 9. Generate predictions
    print("\n=== Generating Predictions ===")
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    # Ensure predictions are float64 for competition precision
    predictions = predictions.astype(np.float64)
    
    print(f"Prediction distribution:")
    print(f"Min: {predictions.min():.6f}")
    print(f"Max: {predictions.max():.6f}")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    
    # 10. Save results
    submission = pd.DataFrame({
        'player_id': test_feat['player_id'],
        'drafted': predictions
    })
    
    # Verify column names
    print(f"\nSubmission file columns: {submission.columns.tolist()}")
    print(f"Submission file first 5 rows:")
    print(submission.head())
    
    submission.to_csv('lightgbm_optuna_predictions.csv', index=False)
    print(f"\nPrediction results saved to lightgbm_optuna_predictions.csv")
    print(f"Submission file shape: {submission.shape}")
    
    # 11. Show model information
    try:
        print(f"\nBest iteration: {model.best_iteration}")
    except:
        print(f"\nNumber of trees used: {model.num_trees()}")
    
    print(f"Best cross-validation AUC: {tuner.best_score:.4f}")
    
    return model, importance_df, predictions

if __name__ == "__main__":
    model, importance, preds = main()