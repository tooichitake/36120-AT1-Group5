"""Feature engineering and selection utilities."""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def create_efficiency_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create efficiency score feature."""
    df = df.copy()
    
    # Use actual column names from the data
    if all(col in df.columns for col in ['eFG', 'ast_tov', 'TS_per']):
        df['efficiency_score'] = (
            df['eFG'] * 0.4 + 
            df['ast_tov'] * 0.3 + 
            df['TS_per'] * 0.3
        )
    elif all(col in df.columns for col in ['eFG', 'TS_per']):
        # Fallback if ast_tov is missing
        df['efficiency_score'] = (
            df['eFG'] * 0.5 + 
            df['TS_per'] * 0.5
        )
    
    return df

def create_all_around_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create all-around player score."""
    df = df.copy()
    
    # Use actual column names from the data
    if all(col in df.columns for col in ['pts', 'treb', 'ast', 'stl', 'blk']):
        df['all_around_score'] = (
            df['pts'] * 0.3 +
            df['treb'] * 0.2 +
            df['ast'] * 0.25 +
            df['stl'] * 0.15 +
            df['blk'] * 0.1
        )
    
    return df

def create_rare_skill_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create indicators for rare skill combinations."""
    df = df.copy()
    
    # Check if necessary columns exist
    if 'type' in df.columns and 'ast' in df.columns and 'treb' in df.columns:
        # Use 'type' column instead of 'pos' if available
        df['versatile_guard'] = (
            (df['type'].isin(['PG', 'SG', 'G', 'Combo G', 'Pure PG', 'Scoring PG'])) & 
            (df['ast'] > df['ast'].quantile(0.7)) & 
            (df['treb'] > df['treb'].quantile(0.7))
        ).astype(int)
    
    if 'type' in df.columns and 'TP_per' in df.columns:
        # Use TP_per (3-point percentage) instead of fg3_pct
        df['stretch_big'] = (
            (df['type'].isin(['PF', 'C', 'F-C', 'C-F', 'Stretch 4', 'Stretch 5'])) & 
            (df['TP_per'] > 0.35)
        ).astype(int)
    
    if 'stl' in df.columns and 'blk' in df.columns:
        df['defensive_specialist'] = (
            (df['stl'] > df['stl'].quantile(0.8)) | 
            (df['blk'] > df['blk'].quantile(0.8))
        ).astype(int)
    
    return df

def create_power_conference_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Create power conference indicator."""
    df = df.copy()
    # Use the actual conference codes from the data
    power_conferences = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']
    if 'conf' in df.columns:
        df['power_conference'] = df['conf'].isin(power_conferences).astype(int)
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Encode categorical features."""
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            df = df.drop(col, axis=1)
    
    return df

def select_features_mutual_info(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int = 30
) -> Tuple[pd.DataFrame, List[str]]:
    """Select top k features using mutual information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

def select_features_anova(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int = 30
) -> Tuple[pd.DataFrame, List[str]]:
    """Select top k features using ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

def normalize_features(X: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normalize features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler

def get_feature_importance_lgb(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Get feature importance from LightGBM model."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    return importance_df

def get_feature_importance_rf(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Get feature importance from Random Forest model."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    return importance_df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between key statistics."""
    df = df.copy()
    
    # Points per minute
    if 'pts' in df.columns and 'mp' in df.columns:
        df['pts_per_min'] = df['pts'] / (df['mp'] + 1e-6)
    
    # Rebounds per minute (use treb for total rebounds)
    if 'treb' in df.columns and 'mp' in df.columns:
        df['reb_per_min'] = df['treb'] / (df['mp'] + 1e-6)
    
    # Assists to points ratio
    if 'ast' in df.columns and 'pts' in df.columns:
        df['ast_pts_ratio'] = df['ast'] / (df['pts'] + 1e-6)
    
    # Defensive rating (blocks + steals)
    if 'blk' in df.columns and 'stl' in df.columns:
        df['defensive_rating'] = df['blk'] + df['stl']
    
    return df

def create_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced basketball metrics."""
    df = df.copy()
    
    # Use BPM (Box Plus/Minus) if available - it's already a good efficiency metric
    if 'bpm' in df.columns:
        df['advanced_metric'] = df['bpm']
    
    # Create a simplified efficiency metric using available columns
    if all(col in df.columns for col in ['pts', 'treb', 'ast', 'stl', 'blk']):
        df['simple_efficiency'] = (
            df['pts'] + df['treb'] + df['ast'] + df['stl'] + df['blk']
        )
        
        # Normalize by games played if available
        if 'GP' in df.columns:
            df['simple_efficiency'] = df['simple_efficiency'] / (df['GP'] + 1e-6)
    
    # True shooting is already in the data as TS_per
    if 'TS_per' in df.columns:
        df['true_shooting'] = df['TS_per']
    
    return df

def create_experiment1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from experiment-1 notebook."""
    df = df.copy()
    
    # Usage efficiency
    if 'usg' in df.columns and 'TS_per' in df.columns:
        df['usage_efficiency'] = (df['usg'] * df['TS_per']).astype(np.float64)
    
    # Minutes impact
    if 'Min_per' in df.columns and 'bpm' in df.columns:
        df['minutes_impact'] = (df['Min_per'] * df['bpm']).astype(np.float64)
    
    # Offensive load
    if 'usg' in df.columns and 'AST_per' in df.columns:
        df['offensive_load'] = (df['usg'] + df['AST_per']).astype(np.float64)
    
    # Assist turnover ratio (use ast_tov if available)
    if 'ast_tov' in df.columns:
        df['assist_turnover_ratio'] = df['ast_tov'].astype(np.float64)
    elif 'AST_per' in df.columns and 'TO_per' in df.columns:
        df['assist_turnover_ratio'] = (df['AST_per'] / (df['TO_per'] + 0.1)).astype(np.float64)
    
    # Defensive stats (use stl_per and blk_per)
    if 'stl_per' in df.columns and 'blk_per' in df.columns:
        df['defensive_stats'] = (df['stl_per'] + df['blk_per']).astype(np.float64)
    
    # Rebound total (use combination of ORB_per and DRB_per)
    if 'ORB_per' in df.columns and 'DRB_per' in df.columns:
        df['rebound_total'] = (df['ORB_per'] + df['DRB_per']).astype(np.float64)
    
    # All-around score (experiment-1 version) - adjusted for available columns
    if all(col in df.columns for col in ['AST_per', 'rebound_total', 'defensive_stats']):
        df['all_around_score_v2'] = (
            df['AST_per'] * 0.3 + 
            df['rebound_total'] * 0.4 + 
            df['defensive_stats'] * 0.3
        ).astype(np.float64)
    
    # True shooting volume
    if 'TS_per' in df.columns and 'usg' in df.columns:
        df['true_shooting_volume'] = (df['TS_per'] * np.log1p(df['usg'])).astype(np.float64)
    
    # Per minute impact
    if 'bpm' in df.columns and 'Min_per' in df.columns:
        df['per_minute_impact'] = (df['bpm'] / (df['Min_per'] + 1)).astype(np.float64)
    
    # Year as numeric
    if 'year' in df.columns:
        df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
    
    return df

def create_advanced_rare_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced rare skill indicators for modern NBA players."""
    df = df.copy()
    
    # Big man shooter (tall players who can shoot 3s)
    if 'height_numeric' in df.columns and 'TP_per' in df.columns:
        df['big_man_shooter'] = ((df['height_numeric'] > 80) & (df['TP_per'] > 35)).astype(int)
    elif 'height_inches' in df.columns and 'TP_per' in df.columns:
        df['big_man_shooter'] = ((df['height_inches'] > 80) & (df['TP_per'] > 35)).astype(int)
    
    # Playmaking big (tall players with high assist rates)
    if 'height_numeric' in df.columns and 'AST_per' in df.columns:
        df['playmaking_big'] = ((df['height_numeric'] > 80) & (df['AST_per'] > 20)).astype(int)
    elif 'height_inches' in df.columns and 'AST_per' in df.columns:
        df['playmaking_big'] = ((df['height_inches'] > 80) & (df['AST_per'] > 20)).astype(int)
    
    # Recruit rank availability indicator
    if 'Rec_Rank' in df.columns:
        df['has_recruit_rank'] = df['Rec_Rank'].notna().astype(int)
    
    return df

def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = create_efficiency_score(df)
    df = create_all_around_score(df)
    df = create_rare_skill_indicators(df)
    df = create_power_conference_indicator(df)
    df = create_interaction_features(df)
    df = create_advanced_metrics(df)
    df = create_experiment1_features(df)
    df = create_advanced_rare_skills(df)
    return df

def lightgbm_feature_importance_analysis(df: pd.DataFrame, target_col: str = 'drafted', 
                                        n_estimators: int = 100, top_n: int = 30):
    """Analyze feature importance using LightGBM quick training."""
    import matplotlib.pyplot as plt
    
    # Prepare data - exclude non-numeric and identifier columns
    exclude_cols = [target_col, 'player_id', 'ht', 'pick']
    X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
    y = df[target_col]
    
    # Handle categorical features
    cat_features = ['team', 'conf', 'yr', 'type']
    cat_features = [f for f in cat_features if f in X.columns]
    
    for cat in cat_features:
        X[cat] = X[cat].astype('category')
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Normalize importance
    importance_df['importance_normalized'] = (
        importance_df['importance'] / importance_df['importance'].sum() * 100
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # 1. Bar plot of importance
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax1.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
    ax1.set_yticks(range(len(importance_df)))
    ax1.set_yticklabels(importance_df['feature'].values)
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title(f'Top {top_n} Features by LightGBM Importance', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add percentage labels
    for bar, val, pct in zip(bars, importance_df['importance'].values, 
                            importance_df['importance_normalized'].values):
        ax1.text(val + 10, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontsize=9)
    
    # 2. Cumulative importance plot
    ax2 = axes[1]
    cumulative_importance = importance_df['importance_normalized'].cumsum()
    ax2.plot(range(len(importance_df)), cumulative_importance.values, 
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.fill_between(range(len(importance_df)), cumulative_importance.values, 
                     alpha=0.3, color='steelblue')
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Cumulative Importance (%)', fontsize=12)
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add 80% line
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Importance')
    ax2.legend()
    
    plt.tight_layout()
    
    print(f"\nTop {top_n} Features by LightGBM Importance:")
    print("="*50)
    print(importance_df[['feature', 'importance', 'importance_normalized']].to_string(index=False))
    
    # Print model performance
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_val)
    print(f"\nValidation Performance:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
    
    return fig, importance_df, model