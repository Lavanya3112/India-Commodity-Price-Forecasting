"""
features.py

Loads the India Master Mandi DB and builds a forecasting-ready dataset.

IMPORTANT DATA NOTE:
Raw records are at (state, district, market, commodity, variety, date) level,
and individual market+commodity combos report very sparsely (median = 1
record). That's realistic for mandi price reporting - not every market
reports every commodity every day. So instead of forecasting per-market
prices, we aggregate to a COMMODITY-LEVEL national daily average price,
which gives dense enough series (400-600+ dates for top commodities) to
actually forecast on.

Columns expected in raw CSV:
    state, district, market, commodity, variety,
    min_price, max_price, modal_price, arrival_date (DD-MM-YYYY)
"""

import pandas as pd
import numpy as np

DATA_PATH = "/home/claude/commodity_price_forecasting/data/India_Master_Mandi_DB.csv"

# Commodities to forecast - top commodities by data coverage (see EDA).
# Feel free to expand this list; anything with 60+ distinct dates works.
TARGET_COMMODITIES = [
    "Wheat", "Potato", "Tomato", "Onion", "Maize",
    "Bengal Gram(Gram)(Whole)", "Banana", "Brinjal",
    "Paddy(Dhan)(Common)", "Mustard",
]

MIN_DATES_REQUIRED = 60  # a commodity needs at least this many distinct
                          # reporting dates to be usable for forecasting


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d-%m-%Y")
    return df


def remove_price_outliers(df: pd.DataFrame, low_ratio: float = 0.1, high_ratio: float = 10.0) -> pd.DataFrame:
    """
    Drops rows where modal_price is a likely data-entry error, e.g. a
    commodity normally trading at Rs 1000-2000/quintal showing up as
    Rs 0.2 - almost certainly a unit mix-up (per-kg entered where
    per-quintal was expected) rather than a real market price.

    Uses each commodity's OWN median as the reference point (not a global
    threshold) since legitimately cheap items like coconut shouldn't be
    flagged just for being inexpensive relative to wheat or cotton.
    """
    df = df.copy()
    medians = df.groupby("commodity")["modal_price"].transform("median")
    is_outlier = (df["modal_price"] < medians * low_ratio) | (df["modal_price"] > medians * high_ratio)
    n_dropped = is_outlier.sum()
    if n_dropped:
        print(f"Dropping {n_dropped} likely data-entry error rows "
              f"(outside {low_ratio}x-{high_ratio}x of each commodity's median price)")
    return df[~is_outlier].reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame, commodities=None) -> pd.DataFrame:
    """
    Collapses raw market-level records into one row per (commodity, date)
    using the mean modal_price across all reporting markets that day.
    This is the national daily average price series per commodity.
    """
    if commodities is not None:
        df = df[df["commodity"].isin(commodities)]

    agg = (
        df.groupby(["commodity", "arrival_date"])
        .agg(
            modal_price=("modal_price", "mean"),
            min_price=("min_price", "mean"),
            max_price=("max_price", "mean"),
            n_markets_reporting=("market", "nunique"),
        )
        .reset_index()
        .sort_values(["commodity", "arrival_date"])
        .reset_index(drop=True)
    )
    return agg


def select_forecastable_commodities(df: pd.DataFrame, min_dates: int = MIN_DATES_REQUIRED) -> list:
    """Returns commodities with enough distinct reporting dates to forecast."""
    counts = df.groupby("commodity")["arrival_date"].nunique()
    return counts[counts >= min_dates].index.tolist()


def build_features(agg_df: pd.DataFrame, target_col: str = "modal_price") -> pd.DataFrame:
    """
    Builds lag, rolling-window, and calendar features per commodity series.
    Note: since reporting dates aren't perfectly regular (gaps of a few days
    are common), lags here are "N reports ago" rather than strictly "N days
    ago". This is a reasonable simplification for a first model; for
    production you'd resample to a regular calendar grid and interpolate.
    """
    df = agg_df.copy()

    df["day_of_week"] = df["arrival_date"].dt.dayofweek
    df["month"] = df["arrival_date"].dt.month
    df["day_of_year"] = df["arrival_date"].dt.dayofyear

    grouped = df.groupby("commodity", group_keys=False)

    for lag in [1, 2, 3, 5, 7]:
        df[f"lag_{lag}"] = grouped[target_col].shift(lag)

    for window in [3, 5, 7]:
        df[f"rolling_mean_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window).std()
        )

    df["pct_change_3"] = grouped[target_col].transform(lambda s: s.shift(1).pct_change(3))
    df["price_range"] = df["max_price"] - df["min_price"]
    df["commodity_code"] = df["commodity"].astype("category").cat.codes

    feature_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "pct_change"))]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def get_feature_columns() -> list:
    return [
        "day_of_week", "month", "day_of_year",
        "lag_1", "lag_2", "lag_3", "lag_5", "lag_7",
        "rolling_mean_3", "rolling_mean_5", "rolling_mean_7",
        "rolling_std_3", "rolling_std_5", "rolling_std_7",
        "pct_change_3", "price_range", "n_markets_reporting",
        "commodity_code",
    ]


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw data: {raw.shape}")
    raw = remove_price_outliers(raw)
    print(f"After removing price outliers: {raw.shape}")

    daily = aggregate_daily(raw)
    usable = select_forecastable_commodities(daily)
    print(f"\nCommodities with >= {MIN_DATES_REQUIRED} distinct dates: {len(usable)}")

    daily = aggregate_daily(raw, commodities=TARGET_COMMODITIES)
    print(f"\nAggregated (target commodities only): {daily.shape}")
    print(daily.groupby("commodity")["arrival_date"].nunique().sort_values(ascending=False))

    feat = build_features(daily)
    print(f"\nAfter feature engineering: {feat.shape}")
    print(feat[["commodity", "arrival_date"] + get_feature_columns() + ["modal_price"]].head())
