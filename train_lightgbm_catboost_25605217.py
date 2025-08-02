import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """加载数据并进行初步分析"""
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 目标变量已经是二分类 (0.0, 1.0)，直接使用
    train_df['drafted_binary'] = train_df['drafted'].astype(int)
    
    print(f"\\n目标分布:")
    drafted_count = train_df['drafted'].sum()
    total_count = len(train_df)
    print(f"被选秀: {int(drafted_count)} ({drafted_count/total_count:.3%})")
    print(f"未被选秀: {total_count - int(drafted_count)} ({(total_count - drafted_count)/total_count:.3%})")
    
    return train_df, test_df

def height_to_inches(height_str):
    """将身高字符串转换为英寸数值"""
    if pd.isna(height_str) or height_str == '-':
        return None  # 返回None，后续用组内均值填充
    if isinstance(height_str, (int, float)):
        return height_str
    
    # 处理 "6-Jun" 格式 (6'6")
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

def advanced_feature_engineering(df, is_train=True):
    """高级特征工程"""
    df_feat = df.copy()
    
    print("\\n=== 开始高级特征工程 ===")
    
    # 1. 身高处理
    df_feat['height_numeric'] = df_feat['ht'].apply(height_to_inches)
    
    # 按位置组填充身高缺失值
    if 'yr' in df_feat.columns:
        df_feat['height_numeric'] = df_feat.groupby(['yr', 'conf'])['height_numeric'].transform(
            lambda x: x.fillna(x.median())
        )
    df_feat['height_numeric'] = df_feat['height_numeric'].fillna(74)  # 总体均值
    
    # 2. 核心实力特征 (基于分析的最重要特征)
    df_feat['is_star_player'] = (df_feat['Min_per'] > 40) & (df_feat['bpm'] > 0)  # 明星球员
    df_feat['usage_efficiency'] = df_feat['usg'] * df_feat['TS_per'] / 100  # 使用率效率
    df_feat['minutes_impact'] = df_feat['Min_per'] * df_feat['bpm']  # 时间加权影响力
    df_feat['offensive_load'] = df_feat['usg'] + df_feat['AST_per']  # 进攻负荷
    
    # 3. 投篮全面性
    df_feat['shooting_versatility'] = (
        (df_feat['TP_per'] > 30).astype(int) +  # 三分能力
        (df_feat['twoP_per'] > 45).astype(int) +  # 两分能力
        (df_feat['FT_per'] > 75).astype(int)   # 罚球能力
    )
    
    # 4. 全能性指标
    df_feat['all_around_ability'] = (
        df_feat['AST_per'] * 0.3 +  # 助攻权重
        (df_feat['ORB_per'] + df_feat['DRB_per']) * 0.4 +  # 篮板权重
        (df_feat['stl_per'] + df_feat['blk_per']) * 0.3   # 防守权重
    )
    
    # 5. 效率vs体量平衡
    df_feat['efficiency_volume_balance'] = df_feat['TS_per'] * np.log1p(df_feat['usg'])
    df_feat['impact_per_minute'] = df_feat['bpm'] / (df_feat['Min_per'] + 1)
    
    # 6. 位置特征 (基于身高和统计)
    df_feat['likely_guard'] = (df_feat['height_numeric'] < 78) & (df_feat['AST_per'] > 15)
    df_feat['likely_center'] = (df_feat['height_numeric'] > 82) & (df_feat['ORB_per'] + df_feat['DRB_per'] > 20)
    df_feat['likely_forward'] = (~df_feat['likely_guard']) & (~df_feat['likely_center'])
    
    # 7. 经验和年龄特征
    yr_to_numeric = {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}
    df_feat['yr_numeric'] = df_feat['yr'].map(yr_to_numeric).fillna(1)
    df_feat['experience_performance'] = df_feat['yr_numeric'] * df_feat['bpm']
    df_feat['is_senior'] = (df_feat['yr'] == 'Sr').astype(int)
    
    # 8. 稀有技能组合
    df_feat['big_man_shooter'] = (df_feat['height_numeric'] > 80) & (df_feat['TP_per'] > 35)
    df_feat['playmaking_big'] = (df_feat['height_numeric'] > 80) & (df_feat['AST_per'] > 20)
    df_feat['defensive_guard'] = (df_feat['height_numeric'] < 78) & (df_feat['stl_per'] > 2.5)
    
    # 9. 会议强度 (基于历史数据的会议等级)
    power_conferences = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']  # 主要会议
    df_feat['power_conference'] = df_feat['conf'].isin(power_conferences).astype(int)
    
    # 10. 招募排名特征 (处理大量缺失值)
    df_feat['has_recruit_rank'] = df_feat['Rec_Rank'].notna().astype(int)
    df_feat['top_recruit'] = (df_feat['Rec_Rank'] <= 100).astype(int)
    df_feat['Rec_Rank_filled'] = df_feat['Rec_Rank'].fillna(999)  # 未排名用999填充
    
    # 11. 进攻vs防守平衡
    df_feat['offense_defense_balance'] = abs(df_feat['obpm'] - df_feat['dbpm'])
    df_feat['total_impact'] = df_feat['obpm'] + df_feat['dbpm']
    
    # 12. 失误控制
    df_feat['turnover_control'] = df_feat['AST_per'] / (df_feat['TO_per'] + 1)  # 助攻失误比
    df_feat['usage_adjusted_turnovers'] = df_feat['TO_per'] / (df_feat['usg'] + 1)
    
    # 13. 身体素质代理指标
    df_feat['athleticism_proxy'] = (
        df_feat['blk_per'] * 0.4 +  # 盖帽能力
        df_feat['stl_per'] * 0.3 +  # 抢断能力  
        df_feat['ORB_per'] * 0.3    # 进攻篮板能力
    )
    
    print(f"特征工程完成，新增特征数量: {len([col for col in df_feat.columns if col not in df.columns])}")
    return df_feat

def prepare_features(train_feat, test_feat):
    """准备最终特征"""
    
    # 选择特征 - 排除ID、目标变量和高缺失率特征
    exclude_cols = [
        'player_id', 'drafted', 'drafted_binary',
        # 高缺失率特征
        'Rec_Rank', 'dunks_ratio', 'rim_ratio', 'mid_ratio', 
        'midmade', 'midmade_midmiss', 'rimmade', 'rimmade_rimmiss',
        'dunksmade', 'dunksmiss_dunksmade',
        # 原始身高列
        'ht'
    ]
    
    # 分类特征
    categorical_features = ['team', 'conf', 'yr', 'type']
    
    # 数值特征
    numerical_features = [col for col in train_feat.columns 
                         if col not in categorical_features + exclude_cols]
    
    print(f"\\n分类特征: {len(categorical_features)}")
    print(f"数值特征: {len(numerical_features)}")
    
    # 编码分类特征
    le_dict = {}
    for cat_feat in categorical_features:
        le = LabelEncoder()
        
        # 获取所有可能的值
        all_values = list(set(
            train_feat[cat_feat].astype(str).unique().tolist() + 
            test_feat[cat_feat].astype(str).unique().tolist()
        ))
        
        le.fit(all_values)
        
        train_feat[cat_feat + '_encoded'] = le.transform(train_feat[cat_feat].astype(str))
        test_feat[cat_feat + '_encoded'] = le.transform(test_feat[cat_feat].astype(str))
        
        le_dict[cat_feat] = le
        print(f"  {cat_feat}: {len(le.classes_)} 个类别")
    
    # 最终特征列
    feature_cols = numerical_features + [f + '_encoded' for f in categorical_features]
    
    # 填充数值特征的缺失值
    for col in numerical_features:
        if train_feat[col].isnull().sum() > 0:
            median_val = train_feat[col].median()
            train_feat[col] = train_feat[col].fillna(median_val)
            test_feat[col] = test_feat[col].fillna(median_val)
    
    X_train = train_feat[feature_cols]
    y_train = train_feat['drafted_binary']
    X_test = test_feat[feature_cols]
    
    print(f"\\n最终特征数量: {len(feature_cols)}")
    print(f"训练集形状: {X_train.shape}")
    
    return X_train, y_train, X_test, feature_cols, le_dict

def objective_catboost(trial, X_train, y_train, feature_cols):
    """CatBoost优化目标函数"""
    params = {
        'iterations': trial.suggest_int('iterations', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 64, 255),
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'class_weights': [1, trial.suggest_int('class_weight_ratio', 50, 150)]  # 处理类别不平衡
    }
    
    cat_features = [i for i, col in enumerate(feature_cols) if col.endswith('_encoded')]
    model = CatBoostClassifier(**params)
    
    # 使用分层交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    
    return scores.mean()

def objective_lightgbm(trial, X_train, y_train):
    """LightGBM优化目标函数"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'random_state': 42,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
        'class_weight': 'balanced'  # 处理类别不平衡
    }
    
    model = lgb.LGBMClassifier(**params)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    
    return scores.mean()

def main():
    """主函数"""
    print("=== NBA选秀预测 - 高级特征工程版本 ===\\n")
    
    # 1. 加载数据
    train_df, test_df = load_and_analyze_data()
    
    # 2. 特征工程
    print("\\n=== 特征工程 ===")
    train_feat = advanced_feature_engineering(train_df, is_train=True)
    test_feat = advanced_feature_engineering(test_df, is_train=False)
    
    # 3. 准备特征
    X_train, y_train, X_test, feature_cols, le_dict = prepare_features(train_feat, test_feat)
    
    # 4. 模型优化
    print("\\n=== 模型优化 ===")
    results = {}
    best_params = {}
    
    # CatBoost
    print("\\n1. 优化 CatBoost...")
    study_cb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_cb.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train, feature_cols), 
        n_trials=300
    )
    results['CatBoost'] = study_cb.best_value
    best_params['CatBoost'] = study_cb.best_params
    print(f"最佳 AUC: {study_cb.best_value:.4f}")
    
    # LightGBM
    print("\\n2. 优化 LightGBM...")
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgb.optimize(
        lambda trial: objective_lightgbm(trial, X_train, y_train),
        n_trials=300
    )
    results['LightGBM'] = study_lgb.best_value
    best_params['LightGBM'] = study_lgb.best_params
    print(f"最佳 AUC: {study_lgb.best_value:.4f}")
    
    # 5. 选择最佳模型并训练
    print("\\n=== 最终模型训练 ===")
    best_model_name = max(results.keys(), key=lambda k: results[k])
    print(f"最佳模型: {best_model_name} (AUC: {results[best_model_name]:.4f})")
    
    if best_model_name == 'CatBoost':
        cat_features = [i for i, col in enumerate(feature_cols) if col.endswith('_encoded')]
        final_model = CatBoostClassifier(**best_params[best_model_name])
        final_model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
    else:
        final_model = lgb.LGBMClassifier(**best_params[best_model_name])
        final_model.fit(X_train, y_train)
    
    # 6. 特征重要性
    print("\\n=== 特征重要性 TOP 20 ===")
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
    else:
        importances = final_model.get_feature_importance()
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    print(feature_importance.to_string(index=False))
    
    # 7. 生成预测
    print("\\n=== 生成预测 ===")
    predictions = final_model.predict_proba(X_test)[:, 1]
    
    print(f"预测分布:")
    print(f"最小值: {predictions.min():.6f}")
    print(f"最大值: {predictions.max():.6f}")
    print(f"平均值: {predictions.mean():.6f}")
    print(f"标准差: {predictions.std():.6f}")
    
    # 8. 保存结果
    submission = pd.DataFrame({
        'id': test_feat['player_id'],
        'drafted': predictions
    })
    
    submission.to_csv('Donald John Trump.csv', index=False)
    print(f"\\n预测结果已保存到 Donald John Trump.csv")
    print(f"提交文件形状: {submission.shape}")
    
    return final_model, feature_importance, predictions

if __name__ == "__main__":
    model, importance, preds = main()