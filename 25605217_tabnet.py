import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# CUDA setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_and_analyze_data():
    """Load data and perform initial analysis"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print(f"\\nTarget distribution:")
    drafted_count = train_df['drafted'].sum()
    total_count = len(train_df)
    print(f"Drafted: {int(drafted_count)} ({drafted_count/total_count:.3%})")
    print(f"Not drafted: {total_count - int(drafted_count)} ({(total_count - drafted_count)/total_count:.3%})")
    
    return train_df, test_df

def height_to_inches(height_str):
    """Convert height string to inches"""
    if pd.isna(height_str) or height_str == '-':
        return None
    if isinstance(height_str, (int, float)):
        return height_str
    
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
            return None
    return None

def advanced_feature_engineering(df):
    """Advanced feature engineering"""
    df_feat = df.copy()
    
    # 1. Height processing
    df_feat['height_numeric'] = df_feat['ht'].apply(height_to_inches)
    
    # Fill missing height values by position group
    if 'yr' in df_feat.columns:
        df_feat['height_numeric'] = df_feat.groupby(['yr', 'conf'])['height_numeric'].transform(
            lambda x: x.fillna(x.median())
        )
    df_feat['height_numeric'] = df_feat['height_numeric'].fillna(74)
    
    # 2. Core performance features (most important features based on analysis)
    df_feat['is_star_player'] = (df_feat['Min_per'] > 40) & (df_feat['bpm'] > 0)
    df_feat['usage_efficiency'] = df_feat['usg'] * df_feat['TS_per'] / 100
    df_feat['minutes_impact'] = df_feat['Min_per'] * df_feat['bpm']
    df_feat['offensive_load'] = df_feat['usg'] + df_feat['AST_per']
    
    # 3. Shooting versatility
    df_feat['shooting_versatility'] = (
        (df_feat['TP_per'] > 30).astype(int) +
        (df_feat['twoP_per'] > 45).astype(int) +
        (df_feat['FT_per'] > 75).astype(int)
    )
    
    # 4. All-around ability metrics
    df_feat['all_around_ability'] = (
        df_feat['AST_per'] * 0.3 +
        (df_feat['ORB_per'] + df_feat['DRB_per']) * 0.4 +
        (df_feat['stl_per'] + df_feat['blk_per']) * 0.3
    )
    
    # 5. Efficiency vs volume balance
    df_feat['efficiency_volume_balance'] = df_feat['TS_per'] * np.log1p(df_feat['usg'])
    df_feat['impact_per_minute'] = df_feat['bpm'] / (df_feat['Min_per'] + 1)
    
    # 6. Position features (based on height and stats)
    df_feat['likely_guard'] = (df_feat['height_numeric'] < 78) & (df_feat['AST_per'] > 15)
    df_feat['likely_center'] = (df_feat['height_numeric'] > 82) & (df_feat['ORB_per'] + df_feat['DRB_per'] > 20)
    df_feat['likely_forward'] = (~df_feat['likely_guard']) & (~df_feat['likely_center'])
    
    # 7. Experience and age features
    yr_to_numeric = {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}
    df_feat['yr_numeric'] = df_feat['yr'].map(yr_to_numeric).fillna(1)
    df_feat['experience_performance'] = df_feat['yr_numeric'] * df_feat['bpm']
    df_feat['is_senior'] = (df_feat['yr'] == 'Sr').astype(int)
    
    # 8. Rare skill combinations
    df_feat['big_man_shooter'] = (df_feat['height_numeric'] > 80) & (df_feat['TP_per'] > 35)
    df_feat['playmaking_big'] = (df_feat['height_numeric'] > 80) & (df_feat['AST_per'] > 20)
    df_feat['defensive_guard'] = (df_feat['height_numeric'] < 78) & (df_feat['stl_per'] > 2.5)
    
    # 9. Conference strength (based on historical conference ranking)
    power_conferences = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']
    df_feat['power_conference'] = df_feat['conf'].isin(power_conferences).astype(int)
    
    # 10. Recruitment ranking features (handling many missing values)
    df_feat['has_recruit_rank'] = df_feat['Rec_Rank'].notna().astype(int)
    df_feat['top_recruit'] = (df_feat['Rec_Rank'] <= 100).astype(int)
    df_feat['Rec_Rank_filled'] = df_feat['Rec_Rank'].fillna(999)
    
    # 11. Offense vs defense balance
    df_feat['offense_defense_balance'] = abs(df_feat['obpm'] - df_feat['dbpm'])
    df_feat['total_impact'] = df_feat['obpm'] + df_feat['dbpm']
    
    # 12. Turnover control
    df_feat['turnover_control'] = df_feat['AST_per'] / (df_feat['TO_per'] + 1)
    df_feat['usage_adjusted_turnovers'] = df_feat['TO_per'] / (df_feat['usg'] + 1)
    
    # 13. Athleticism proxy indicators
    df_feat['athleticism_proxy'] = (
        df_feat['blk_per'] * 0.4 +
        df_feat['stl_per'] * 0.3 +  
        df_feat['ORB_per'] * 0.3
    )
    
    return df_feat

def prepare_features(train_feat, test_feat):
    """Prepare final features"""
    
    exclude_cols = [
        'player_id', 'drafted',
        'Rec_Rank', 'dunks_ratio', 'rim_ratio', 'mid_ratio', 
        'midmade', 'midmade_midmiss', 'rimmade', 'rimmade_rimmiss',
        'dunksmade', 'dunksmiss_dunksmade',
        'ht'
    ]
    
    categorical_features = ['team', 'conf', 'yr', 'type']
    
    numerical_features = [col for col in train_feat.columns 
                         if col not in categorical_features + exclude_cols]
    
    print(f"\\nCategorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    
    for cat_feat in categorical_features:
        le = LabelEncoder()
        
        all_values = list(set(
            train_feat[cat_feat].astype(str).unique().tolist() + 
            test_feat[cat_feat].astype(str).unique().tolist()
        ))
        
        le.fit(all_values)
        
        train_feat[cat_feat + '_encoded'] = le.transform(train_feat[cat_feat].astype(str))
        test_feat[cat_feat + '_encoded'] = le.transform(test_feat[cat_feat].astype(str))
        
    
    feature_cols = numerical_features + [f + '_encoded' for f in categorical_features]
    
    for col in numerical_features:
        if train_feat[col].isnull().sum() > 0:
            median_val = train_feat[col].median()
            train_feat[col] = train_feat[col].fillna(median_val)
            test_feat[col] = test_feat[col].fillna(median_val)
    
    X_train = train_feat[feature_cols]
    y_train = train_feat['drafted'].astype(int)
    X_test = test_feat[feature_cols]
    
    print(f"\\nFinal feature count: {len(feature_cols)}")
    print(f"Training set shape: {X_train.shape}")
    
    return X_train, y_train, X_test, feature_cols

def objective_tabnet(trial, X_train, y_train):
    """TabNet optimization objective function - using AdamW with wide parameter ranges"""
    batch_size = 128
    virtual_batch_size = 128
    
    torch.set_default_dtype(torch.float32)
    X_train = X_train.astype(np.float32)
    
    params = {
        'n_d': trial.suggest_int('n_d', 16, 128),
        'n_a': trial.suggest_int('n_a', 16, 128),
        'n_steps': trial.suggest_int('n_steps', 5, 15),
        'gamma': trial.suggest_float('gamma', 1.2, 2.5),
        'n_independent': trial.suggest_int('n_independent', 1, 5),  
        'n_shared': trial.suggest_int('n_shared', 1, 5),  
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),
        'momentum': trial.suggest_float('momentum', 0.02, 0.4),  
        'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-3, log=True),  
        'optimizer_fn': torch.optim.AdamW,
        'optimizer_params': dict(
            lr=trial.suggest_float('lr', 5e-4, 5e-2, log=True), 
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        ),
        'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        'scheduler_params': {"step_size": 20, "gamma": 0.8},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'seed': 42,
        'verbose': 0,
        'device_name': device  
    }
    
    
    # Select scaling method
    scaling_method = trial.suggest_categorical('scaling_method', ['standard', 'minmax', 'robust'])
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:  # 'robust'
        scaler = RobustScaler()
    
    model = TabNetClassifier(**params)
    
    # Use stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train = scaler.fit_transform(X_train.iloc[train_idx]).astype(np.float32)
        X_fold_val = scaler.transform(X_train.iloc[val_idx]).astype(np.float32)
        
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pos_weight = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()
        
        use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
        if use_focal_loss:
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 3.0)
            # Create sample weights for focal loss
            # For binary classification, focal loss weight is (1-p_t)^gamma
            # Simplified here, using power of class imbalance weights
            weights = (1 + pos_weight * y_fold_train.values) ** focal_gamma
        else:
            weights = 1 + pos_weight * y_fold_train.values  # Handle class imbalance
        
        model.fit(
            X_fold_train, y_fold_train.values,
            eval_set=[(X_fold_val, y_fold_val.values)],
            max_epochs=150,
            patience=15,
            batch_size=batch_size,  
            virtual_batch_size=virtual_batch_size, 
            num_workers=0,
            drop_last=False,
            weights=weights
        )
        
        y_pred = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

def main():
    """Main function"""
    print("=== NBA Draft Prediction - TabNet Deep Learning Version ===\\n")
    
    # 1. Load data
    train_df, test_df = load_and_analyze_data()
    
    # 2. Feature engineering
    print("\\n=== Feature Engineering ===")
    train_feat = advanced_feature_engineering(train_df)
    test_feat = advanced_feature_engineering(test_df)
    print("Feature engineering completed")
    
    # 3. Prepare features
    X_train, y_train, X_test, feature_cols = prepare_features(train_feat, test_feat)
    
    # 4. Model optimization
    print("\\n=== Model Optimization ===")
    
    # TabNet optimization
    print("\\nOptimizing TabNet...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_tabnet(trial, X_train, y_train), 
        n_trials=50  
    )
    
    best_score = study.best_value
    best_params = study.best_params
    print(f"Best AUC: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # 5. Final model training
    print("\\n=== Final Model Training ===")
    
    best_scaling_method = best_params.get('scaling_method', 'standard')
    print(f"Using scaling method: {best_scaling_method}")
    
    torch.set_default_dtype(torch.float32)
    
    if best_scaling_method == 'standard':
        scaler = StandardScaler()
    elif best_scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:  # 'robust'
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    final_params = {}
    
    for key in ['n_d', 'n_a', 'n_steps', 'gamma', 'n_independent', 'n_shared', 
                'lambda_sparse', 'momentum', 'epsilon', 'mask_type']:
        if key in best_params:
            final_params[key] = best_params[key]
    
    
    final_params['optimizer_fn'] = torch.optim.AdamW
    final_params['optimizer_params'] = dict(
        lr=best_params.get('lr', 1e-3),
        weight_decay=best_params.get('weight_decay', 1e-5)
    )
    
    final_params['scheduler_params'] = {"step_size": 20, "gamma": 0.8}
    final_params['scheduler_fn'] = torch.optim.lr_scheduler.StepLR
    final_params['seed'] = 42
    final_params['verbose'] = 0
    final_params['device_name'] = device
    
    if best_params.get('use_focal_loss', False):
        focal_gamma = best_params.get('focal_gamma', 2.0)
        weights = (1 + pos_weight * y_train.values) ** focal_gamma
    else:
        weights = 1 + pos_weight * y_train.values
    
    final_model = TabNetClassifier(**final_params)
    final_model.fit(
        X_train_scaled, y_train.values,
        max_epochs=200,
        patience=20,
        batch_size=128,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=weights
    )
    
    # 6. Feature importance
    print("\\n=== Feature Importance TOP 20 ===")
    try:
        importances = final_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)
        print(feature_importance.to_string(index=False))
    except Exception as e:
        feature_importance = pd.DataFrame()
        print(f"Unable to get feature importance: {e}")
    
    # 7. Generate predictions
    print("\\n=== Generating Predictions ===")
    predictions = final_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Prediction distribution:")
    print(f"Minimum: {predictions.min():.6f}")
    print(f"Maximum: {predictions.max():.6f}")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Standard deviation: {predictions.std():.6f}")
    
    submission = pd.DataFrame({
        'id': test_df['player_id'],  
        'drafted': predictions
    })
    
    submission.to_csv('Donald_John_Trump.csv', index=False)
    print(f"\\nPrediction results saved to Donald John Trump.csv")
    print(f"Submission file shape: {submission.shape}")
    
    return final_model, feature_importance, predictions

if __name__ == "__main__":
    model, importance, preds = main()