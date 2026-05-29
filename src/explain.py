import os
import logging
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_explanation(df_features: pd.DataFrame, config: dict, sample_size: int = 1000):
    """
    Генерация объяснений SHAP и сохранение графиков важности признаков.
    """
    logger.info("Инициализация SHAP Explainer...")

    # 1. Загрузка модели
    model_path = os.path.join(config['paths']['model_dir'], "lgbm_master.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена по пути {model_path}. Сначала запустите обучение.")
        return

    model = joblib.load(model_path)

    # 2. Подготовка данных (берем только признаки)
    exclude_cols = [config['pipeline']['target_col'], config['pipeline']['group_col'], "bank_name"]
    features = [c for c in df_features.columns if c not in exclude_cols]

    # Для SHAP берем случайную подвыборку, так как расчет на 105k строках может быть долгим
    X_sample = df_features[features].sample(n=min(sample_size, len(df_features)), random_state=42)

    # 3. Вычисление SHAP значений
    # В LightGBM для бинарной классификации SHAP может возвращать список массивов.
    # Нас интересует класс 1 (Бизнес).
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Обработка формата SHAP (зависит от версии shap/lightgbm)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]  # Берем вероятности для класса 1
    else:
        shap_values_to_plot = shap_values

    # 4. Сохранение визуализации
    plot_dir = config['paths']['log_dir']
    os.makedirs(plot_dir, exist_ok=True)

    logger.info("Генерация графика SHAP Summary Plot...")

    plt.figure(figsize=(10, 6))
    # SHAP генерирует график напрямую через matplotlib
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)

    summary_path = os.path.join(plot_dir, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    plt.close()

    logger.info(f"График важности признаков сохранен в {summary_path}")

    return explainer, shap_values