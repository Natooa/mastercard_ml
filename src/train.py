import os
import logging
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

logger = logging.getLogger(__name__)


def train_model(df_features: pd.DataFrame, config: dict):
    logger.info("Preparing data for training...")
    target_col = config['pipeline']['target_col']
    group_col = config['pipeline']['group_col']

    # Exclude metadata from features
    exclude_cols = [target_col, group_col, "bank_name"]
    features = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[features]
    y = df_features[target_col]
    groups = df_features[group_col]

    params = config['model_params']
    params['random_state'] = config['pipeline']['random_state']
    n_estimators = params.pop('n_estimators')

    cv = StratifiedGroupKFold(n_splits=5)
    y_prob_cv = np.zeros(len(y))

    models = []

    logger.info("Starting Cross-Validation training...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        y_prob_cv[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        models.append(model)
        logger.info(f"Fold {fold + 1} completed.")

    auc = roc_auc_score(y, y_prob_cv)
    logger.info(f"Final OOF ROC-AUC: {auc:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y, y_prob_cv)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_thresh = thresholds[np.argmax(f1_scores)]
    logger.info(f"Best classification threshold (F1-optimized): {best_thresh:.4f}")

    # Retrain on full data for production model
    logger.info("Retraining final model on full dataset...")
    final_model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
    final_model.fit(X, y)

    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    model_path = os.path.join(config['paths']['model_dir'], "lgbm_master.joblib")
    joblib.dump(final_model, model_path)
    logger.info(f"Model serialized and saved to {model_path}")

    return final_model