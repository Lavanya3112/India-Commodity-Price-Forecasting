"""
train.py

Trains and compares three forecasting models on the commodity price data:
  1. Naive baseline    - predicts "same as last known price" (sanity check)
  2. Random Forest     - simple, interpretable, hard to beat on tabular data
  3. LightGBM          - usually best accuracy for this kind of feature set

Uses a CHRONOLOGICAL train/test split (never random!) - the model is
evaluated on data that comes AFTER the training period, which is the only
honest way to test a forecasting model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

from features import (
    load_raw, remove_price_outliers, aggregate_daily,
    build_features, get_feature_columns, TARGET_COMMODITIES
)

TEST_FRACTION = 0.2  # last 20% of dates (per commodity) held out for testing


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """
    Splits each commodity's series separately by date, so every commodity
    contributes both train and test rows, and no future data leaks into
    training.
    """
    train_parts, test_parts = [], []
    for commodity, group in df.groupby("commodity"):
        group = group.sort_values("arrival_date")
        split_idx = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    # Guard against near-zero prices blowing up MAPE (division instability);
    # outlier removal in features.py should already prevent most of this.
    safe_true = np.where(np.abs(y_true) < 1, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / safe_true)) * 100
    print(f"{label:20s} | MAE: {mae:8.2f}  RMSE: {rmse:8.2f}  MAPE: {mape:6.2f}%")
    return {"label": label, "mae": mae, "rmse": rmse, "mape": mape}


def naive_baseline_predict(test_df: pd.DataFrame) -> np.ndarray:
    """Predicts tomorrow's price = last known price (lag_1). The bar every
    real model must clear."""
    return test_df["lag_1"].values


def main():
    print("Loading and aggregating data...")
    raw = load_raw()
    raw = remove_price_outliers(raw)
    daily = aggregate_daily(raw, commodities=TARGET_COMMODITIES)
    feat_df = build_features(daily)
    feature_cols = get_feature_columns()

    train_df, test_df = chronological_split(feat_df)
    print(f"\nTrain rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print(f"Train date range: {train_df.arrival_date.min().date()} to {train_df.arrival_date.max().date()}")
    print(f"Test date range:  {test_df.arrival_date.min().date()} to {test_df.arrival_date.max().date()}\n")

    X_train, y_train = train_df[feature_cols], train_df["modal_price"]
    X_test, y_test = test_df[feature_cols], test_df["modal_price"]

    results = []

    # 1. Naive baseline
    naive_preds = naive_baseline_predict(test_df)
    results.append(evaluate(y_test.values, naive_preds, "Naive (lag-1)"))

    # 2. Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate(y_test.values, rf_preds, "Random Forest"))

    # 3. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        num_leaves=31, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    results.append(evaluate(y_test.values, lgb_preds, "LightGBM"))

    # Save everything needed for evaluation/plots
    joblib.dump(rf, "/home/claude/commodity_price_forecasting/outputs/rf_model.pkl")
    joblib.dump(lgb_model, "/home/claude/commodity_price_forecasting/outputs/lgb_model.pkl")
    test_df = test_df.copy()
    test_df["naive_pred"] = naive_preds
    test_df["rf_pred"] = rf_preds
    test_df["lgb_pred"] = lgb_preds
    test_df.to_csv("/home/claude/commodity_price_forecasting/outputs/test_predictions.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv("/home/claude/commodity_price_forecasting/outputs/model_comparison.csv", index=False)

    print("\nFeature importance (LightGBM, top 8):")
    importance = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importance.head(8))

    print("\nSaved: outputs/rf_model.pkl, outputs/lgb_model.pkl, "
          "outputs/test_predictions.csv, outputs/model_comparison.csv")

    return results_df


if __name__ == "__main__":
    main()
