from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def train_xgboost(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def train_lightgbm(X, y):
    model = LGBMClassifier()
    model.fit(X, y)
    return model

def evaluate_model(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)
