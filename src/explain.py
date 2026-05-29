import os
import logging
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_explanation(df_features: pd.DataFrame, config: dict, sample_size: int = 1000):
    logger.info("Инициализация SHAP Explainer...")

    model_path = os.path.join(config['paths']['model_dir'], "lgbm_master.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена по пути {model_path}. Сначала запустите обучение.")
        return

    # 1. Загружаем артефакт — теперь это словарь {"model": ..., "features": ...}
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]  # точно тот же список, что при обучении

    # 2. Берём только нужные признаки (без пересчёта exclude_cols)
    X_sample = df_features[features].sample(n=min(sample_size, len(df_features)), random_state=42)

    # 3. SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # 4. Сохранение
    plot_dir = config['paths']['log_dir']
    os.makedirs(plot_dir, exist_ok=True)

    logger.info("Генерация графика SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)

    summary_path = os.path.join(plot_dir, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    plt.close()

    logger.info(f"График важности признаков сохранен в {summary_path}")
    return explainer, shap_values