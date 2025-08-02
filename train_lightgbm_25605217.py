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
    df_feat['height_numeric'] = df_feat['ht'].apply(height_to_inches)
    
    # 2. Basic interaction features
    df_feat['usage_efficiency'] = df_feat['usg'] * df_feat['TS_per']
    df_feat['minutes_impact'] = df_feat['Min_per'] * df_feat['bpm']
    df_feat['offensive_load'] = df_feat['usg'] + df_feat['AST_per']
    
    # 3. Ratio features
    df_feat['assist_turnover_ratio'] = df_feat['AST_per'] / (df_feat['TO_per'] + 0.1)
    df_feat['rebound_total'] = df_feat['ORB_per'] + df_feat['DRB_per']
    df_feat['defensive_stats'] = df_feat['stl_per'] + df_feat['blk_per']
    
    # 4. Advanced features
    df_feat['all_around_score'] = (
        df_feat['AST_per'] * 0.3 + 
        df_feat['rebound_total'] * 0.4 + 
        df_feat['defensive_stats'] * 0.3
    )
    
    # 5. Efficiency metrics
    df_feat['true_shooting_volume'] = df_feat['TS_per'] * np.log1p(df_feat['usg'])
    df_feat['per_minute_impact'] = df_feat['bpm'] / (df_feat['Min_per'] + 1)
    
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
    
    # 选择特征 - 排除ID和目标变量
    exclude_cols = ['player_id', 'drafted', 'ht']
    
    # 分类特征
    categorical_features = ['team', 'conf', 'yr', 'type']
    
    # 处理分类特征 - LightGBM支持直接使用类别型
    for cat_feat in categorical_features:
        train_feat[cat_feat] = train_feat[cat_feat].astype('category')
        test_feat[cat_feat] = test_feat[cat_feat].astype('category')
        
        # 确保训练集和测试集有相同的类别
        all_cats = set(train_feat[cat_feat].cat.categories) | set(test_feat[cat_feat].cat.categories)
        train_feat[cat_feat] = train_feat[cat_feat].cat.set_categories(all_cats)
        test_feat[cat_feat] = test_feat[cat_feat].cat.set_categories(all_cats)
    
    # 所有特征
    feature_cols = [col for col in train_feat.columns if col not in exclude_cols]
    
    X_train = train_feat[feature_cols]
    y_train = train_feat['drafted']
    X_test = test_feat[feature_cols]
    
    print(f"\n最终特征数量: {len(feature_cols)}")
    print(f"分类特征: {categorical_features}")
    print(f"训练集形状: {X_train.shape}")
    
    return X_train, y_train, X_test, feature_cols, categorical_features

def main():
    """Main function"""
    print("=== NBA Draft Prediction - LightGBM Optuna Native Integration ===\n")
    
    # 1. 加载数据
    train_df, test_df = load_and_analyze_data()
    
    # 2. 特征工程 - 保留缺失值
    print("\n=== Feature Engineering ===")
    train_feat = create_features_for_lightgbm(train_df)
    test_feat = create_features_for_lightgbm(test_df)
    
    # 3. 准备数据集
    X_train, y_train, X_test, feature_cols, categorical_features = prepare_lightgbm_datasets(train_feat, test_feat)
    
    # 4. 设置基础参数（只包含必要的固定参数）
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'dart',  # 对于表格数据最稳定可靠,gbdt,dart也可以尝试
        'num_threads': -1,
        'verbosity': -1,
    }
    
    # 5. 创建数据集
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    
    # 6. 使用Optuna的LightGBM集成进行超参数优化
    print("\n=== LightGBM Optuna Native Optimization ===")
    print("Optuna will automatically search for best hyperparameters...")
    
    # 使用optuna.integration.lightgbm的LightGBMTunerCV进行优化
    tuner = lgb_optuna.LightGBMTunerCV(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=1000,
        nfold=5,  # 5折交叉验证
        stratified=True,  # 分层采样（对于分类问题）
        shuffle=True,  # 打乱数据
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0)  # 不显示训练日志
        ],
        show_progress_bar=False,  # 不显示进度条以保持输出清洁
        return_cvbooster=True  # 允许获取最佳模型
    )
    
    # 运行优化
    print("Starting hyperparameter optimization (this may take several minutes)...")
    print("Optimization progress:")
    tuner.run()
    print("✓ Hyperparameter optimization completed")
    
    # 获取最佳参数
    best_params = tuner.best_params
    best_params.update(lgb_params)
    
    print("\n=== Best Parameters ===")
    for key, value in best_params.items():
        if key not in lgb_params:  # 只显示优化的参数
            print(f"  {key}: {value}")
    
    # 获取最佳模型（CVBooster）
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
    
    # 7. 显示最佳分数
    print(f"\nBest cross-validation AUC: {tuner.best_score:.4f}")
    
    # 8. 特征重要性
    print("\n=== Top 20 Feature Importance ===")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    
    print(importance_df.to_string(index=False))
    
    # 9. 生成预测
    print("\n=== Generating Predictions ===")
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    print(f"Prediction distribution:")
    print(f"Min: {predictions.min():.6f}")
    print(f"Max: {predictions.max():.6f}")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    
    # 10. 保存结果
    submission = pd.DataFrame({
        'id': test_feat['player_id'],
        'drafted': predictions
    })
    
    # 验证列名
    print(f"\nSubmission file columns: {submission.columns.tolist()}")
    print(f"Submission file first 5 rows:")
    print(submission.head())
    
    submission.to_csv('lightgbm_optuna_predictions.csv', index=False)
    print(f"\nPrediction results saved to lightgbm_optuna_predictions.csv")
    print(f"Submission file shape: {submission.shape}")
    
    # 11. 显示模型信息
    try:
        print(f"\nBest iteration: {model.best_iteration}")
    except:
        print(f"\nNumber of trees used: {model.num_trees()}")
    
    print(f"Best cross-validation AUC: {tuner.best_score:.4f}")
    
    return model, importance_df, predictions

if __name__ == "__main__":
    model, importance, preds = main()