import argparse
import yaml
from src.logger import setup_logger
from src.features import build_features
from src.train import train_model
from src.explain import run_explanation


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mastercard Transaction Classifier Pipeline")
    # ИСПРАВЛЕНИЕ 1: Добавлен "explain" в список choices
    parser.add_argument("--step", choices=["all", "features", "train", "explain"], default="all",
                        help="Pipeline step to run")
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger(config['paths']['log_dir'])

    logger.info(f"Starting pipeline execution. Step: {args.step}")

    try:
        # Блок генерации признаков
        if args.step in ["all", "features"]:
            df_features = build_features(config)

        # Блок обучения модели
        if args.step in ["all", "train"]:
            if args.step == "train":
                import pandas as pd

                feat_path = f"{config['paths']['processed_data_dir']}/{config['paths']['output_features']}"
                df_features = pd.read_parquet(feat_path)

            # Convert to Pandas for sklearn compatibility
            df_pandas = df_features.to_pandas() if hasattr(df_features, 'to_pandas') else df_features
            model = train_model(df_pandas, config)

        # ИСПРАВЛЕНИЕ 2: Добавлен блок генерации объяснений SHAP
        if args.step in ["all", "explain"]:
            if args.step == "explain":
                import pandas as pd

                feat_path = f"{config['paths']['processed_data_dir']}/{config['paths']['output_features']}"
                df_features = pd.read_parquet(feat_path)

            # Convert to Pandas for SHAP compatibility
            df_pandas = df_features.to_pandas() if hasattr(df_features, 'to_pandas') else df_features
            run_explanation(df_pandas, config)

        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)