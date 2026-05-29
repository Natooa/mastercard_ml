import os
import polars as pl
import logging

logger = logging.getLogger(__name__)


def build_features(config: dict) -> pl.DataFrame:
    """High-performance feature aggregation using Polars."""
    b_path = os.path.join(config['paths']['raw_data_dir'], config['paths']['business_file'])
    c_path = os.path.join(config['paths']['raw_data_dir'], config['paths']['consumer_file'])

    logger.info("Initializing lazy queries for raw Parquet files...")
    cols_to_read = [
        "card_number", "transaction_timestamp", "transaction_amount_kzt",
        "mcc", "merchant_id", "channel", "country", "tokenized", "is_recurring", "bank_name"
    ]

    q_b = pl.scan_parquet(b_path).select(cols_to_read).with_columns(pl.lit(1, dtype=pl.Int8).alias("target"))
    q_c = pl.scan_parquet(c_path).select(cols_to_read).with_columns(pl.lit(0, dtype=pl.Int8).alias("target"))

    q_all = pl.concat([q_b, q_c])

    q_all = q_all.with_columns([
        pl.col("card_number").cast(pl.Categorical),
        pl.col("mcc").cast(pl.Categorical),
        pl.col("merchant_id").cast(pl.Categorical)
    ]).sort(["card_number", "transaction_timestamp"])

    logger.info("Extracting temporal and structural components...")
    q_all = q_all.with_columns([
        pl.col("transaction_timestamp").dt.hour().alias("hour"),
        pl.col("transaction_timestamp").dt.weekday().alias("weekday"),
        pl.col("transaction_timestamp").dt.date().alias("date"),
        (pl.col("country") != "Kazakhstan").alias("is_foreign"),
        (pl.col("channel") == "online").alias("is_online"),
        pl.col("transaction_timestamp").diff().dt.total_minutes().over("card_number").alias("time_diff_mins")
    ])

    daily_stats = q_all.group_by(["card_number", "date"]).agg([
        pl.col("transaction_amount_kzt").sum().alias("daily_amount"),
        pl.len().alias("daily_tx_count")
    ]).group_by("card_number").agg([
        pl.col("daily_amount").max().alias("max_daily_amount"),
        pl.col("daily_amount").mean().alias("mean_daily_amount"),
        pl.col("daily_tx_count").max().alias("max_tx_per_day"),
        pl.col("daily_tx_count").std().alias("daily_tx_count_std"),
        pl.len().alias("active_days")
    ])

    logger.info("Aggregating macro features per card...")
    main_features = q_all.group_by("card_number").agg([
        pl.col("target").first().alias("target"),
        pl.len().alias("tx_count"),
        pl.col("transaction_amount_kzt").sum().alias("amount_sum"),
        pl.col("transaction_amount_kzt").mean().alias("amount_mean"),
        pl.col("transaction_amount_kzt").max().alias("amount_max"),
        pl.col("transaction_amount_kzt").std().alias("amount_std"),
        pl.col("merchant_id").n_unique().alias("merchant_nunique"),
        pl.col("mcc").n_unique().alias("mcc_nunique"),
        (pl.col("mcc").is_in(['7311']).sum() / pl.len()).alias("mcc_7311_share"),
        ((pl.col("hour") < 6).sum() / pl.len()).alias("night_tx_share"),
        (pl.col("is_recurring").cast(pl.Int32).sum() / pl.len()).alias("recurring_share"),
        pl.col("time_diff_mins").mean().alias("mean_time_diff_mins")
    ])

    final_lf = main_features.join(daily_stats, on="card_number", how="left").with_columns([
        (pl.col("mcc_nunique") / pl.col("tx_count")).alias("mcc_diversity_index"),
        (pl.col("amount_sum") / pl.col("active_days")).alias("spend_velocity_per_active_day")
    ]).fill_nan(0).fill_null(0)

    logger.info("Executing streaming collection...")
    df = final_lf.collect(streaming=True)

    out_path = os.path.join(config['paths']['processed_data_dir'], config['paths']['output_features'])
    os.makedirs(config['paths']['processed_data_dir'], exist_ok=True)
    df.write_parquet(out_path)
    logger.info(f"Features saved to {out_path}. Unique cards processed: {df.shape[0]}")
    return df