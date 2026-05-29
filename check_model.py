import joblib
import pandas as pd
import yaml

# 1. Загружаем конфиг и пути
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

feat_path = f"{config['paths']['processed_data_dir']}/{config['paths']['output_features']}"
model_path = f"{config['paths']['model_dir']}/lgbm_master.joblib"

# 2. Загружаем данные и модель
print("Загрузка данных и модели...")
df = pd.read_parquet(feat_path)
model = joblib.load(model_path)

# 3. Подготавливаем признаки (убираем метаданные, как при обучении)
exclude_cols = [config['pipeline']['target_col'], config['pipeline']['group_col'], "bank_name"]
features = [c for c in df.columns if c not in exclude_cols]
X = df[features]
y_true = df[config['pipeline']['target_col']]

# ---------------------------------------------------------
# ПРОВЕРКА №1: Важность признаков
# ---------------------------------------------------------
print("\n=== ТОП-10 ВАЖНЫХ ПРИЗНАКОВ ===")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance.head(10).to_string(index=False))

# ---------------------------------------------------------
# ПРОВЕРКА №2: Предсказание на случайных картах
# ---------------------------------------------------------
print("\n=== ТЕСТОВЫЕ ПРЕДСКАЗАНИЯ (5 случайных карт) ===")
# Берем 5 случайных записей
sample_df = df.sample(5, random_state=42)
X_sample = sample_df[features]

# Получаем вероятности (вероятность класса 1 - Бизнес)
probabilities = model.predict_proba(X_sample)[:, 1]

results = pd.DataFrame({
    'Card_Number': sample_df[config['pipeline']['group_col']],
    'Real_Target (1=Бизнес)': sample_df[config['pipeline']['target_col']],
    'Predicted_Probability': probabilities
})

# Форматируем вероятность в проценты для удобства
results['Predicted_Probability'] = results['Predicted_Probability'].apply(lambda x: f"{x:.2%}")
print(results.to_string(index=False))