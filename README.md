# India Commodity Price Forecasting

A beginner-friendly ML project that forecasts daily average mandi prices for
Indian agricultural commodities, built on the `India_Master_Mandi_DB.csv`
dataset (state, district, market, commodity, variety, min/max/modal price,
arrival date).

This is the natural "predictive layer" companion to a dashboard's Price
Spike Alert / Market Recommendation Engine - instead of just reacting to
past prices, it estimates where prices are headed next.

## Project structure

```
commodity_price_forecasting/
├── data/
│   └── India_Master_Mandi_DB.csv     # raw dataset
├── src/
│   ├── features.py                   # data loading, cleaning, feature engineering
│   ├── train.py                      # trains naive/RF/LightGBM models
│   └── evaluate.py                   # generates comparison plots
├── outputs/
│   ├── model_comparison.csv/.png
│   ├── forecast_plots.png
│   ├── test_predictions.csv
│   ├── rf_model.pkl
│   └── lgb_model.pkl
└── requirements.txt
```

## Key data finding (important - read this first)

Raw records are at the individual (state, market, commodity, date) level,
and **each exact combination reports very sparsely** - the median is just
1 record per combination across the whole ~4-year dataset. This is
realistic: not every mandi reports every commodity every day.

This means per-market forecasting (e.g. "predict tomorrow's onion price in
Nashik APMC specifically") isn't viable with this data - there simply
aren't enough repeated observations per exact series.

**The fix:** aggregate to a national daily average price per commodity
(mean `modal_price` across all markets reporting that commodity on a given
day). This gives dense, forecastable series - the top commodities
(Wheat, Potato, Tomato, Onion, Maize, etc.) have 400-600+ distinct
reporting dates over the dataset's span. Ten such commodities are
forecast by default (see `TARGET_COMMODITIES` in `features.py`); any
commodity with 60+ distinct dates can be added.

## Second data finding: outlier prices

A small number of rows (603 out of ~91k, about 0.7%) have modal prices
that are 10x+ away from that commodity's own median - e.g. Banana at
Rs 0.225 or Lemon at Rs 0.4, when these commodities normally trade in the
hundreds to thousands per quintal. These are almost certainly unit
mix-ups (e.g. per-kg price entered where per-quintal was expected) rather
than real market prices, and they badly distort both training and
evaluation (a single Rs 0.225 actual price sends MAPE into the thousands
of percent through division). `remove_price_outliers()` in `features.py`
filters these using each commodity's own median as the reference point,
so genuinely cheap items (like coconut) aren't wrongly flagged.

**Takeaway for you:** always sanity-check target values before training,
and always check whether an evaluation metric (MAPE especially) can blow
up on edge cases before trusting it.

## Approach

1. **Load & clean** - parse dates, drop data-entry price outliers.
2. **Aggregate** - one row per (commodity, date), national mean price.
3. **Feature engineer**, per commodity:
   - Lag features: price 1, 2, 3, 5, 7 reports ago (the most predictive
     features in price forecasting - "what was it recently" beats almost
     everything else)
   - Rolling mean/std over 3, 5, 7 reports (recent trend + volatility)
   - Percent change over 3 reports (momentum)
   - Calendar features: day of week, month, day of year (seasonality)
   - Price range (max - min) as a proxy for market volatility that day
4. **Chronological train/test split** - the LAST 20% of dates per
   commodity are held out as test data. This is critical: a random split
   would leak future information into training and give a falsely
   optimistic result. Forecasting models must always be evaluated on data
   that comes after the training period.
5. **Compare three models:**
   - **Naive baseline** (predict tomorrow = today) - always build this
     first; it's your sanity check. Any real model that can't beat it
     isn't adding value.
   - **Random Forest** - simple, robust, surprisingly hard to beat on
     tabular data like this.
   - **LightGBM** - typically the best accuracy/effort tradeoff for
     tabular time series with engineered features.

## Results

| Model         | MAE (Rs) | RMSE (Rs) | MAPE   |
|---------------|---------:|----------:|-------:|
| Naive (lag-1) |   520.26 |    779.43 | 23.97% |
| Random Forest |   407.20 |    592.40 | 19.17% |
| LightGBM      |   414.37 |    600.43 | 19.13% |

Both real models beat the naive baseline by roughly 20% on MAE/RMSE -
a solid result for a first pass. Looking at `forecast_plots.png`, the
models track the overall price level and trend well but smooth over sharp
day-to-day spikes (very visible for Tomato and Maize, which are the most
volatile commodities in the set). That's expected: lag/rolling features
inherently lag behind sudden shocks. A dedicated spike-detection model
(classification: "will there be a spike tomorrow, yes/no") would be the
natural next project to pair with this one.

## How to run

```bash
pip install -r requirements.txt
cd src
python train.py       # trains models, saves to ../outputs/
python evaluate.py    # generates plots from ../outputs/
```

## Where to go next (in rough order of difficulty)

1. **Add more commodities** - lower `MIN_DATES_REQUIRED` in `features.py`
   and expand `TARGET_COMMODITIES`, or forecast all 131 commodities that
   clear the 60-date bar in a loop.
2. **Hyperparameter tuning** - use `GridSearchCV` or `Optuna` on the
   LightGBM params (currently hand-picked, not tuned).
3. **State-level forecasts** - instead of national average, forecast per
   (commodity, state) for the combos with enough density (213 combos have
   60+ dates per the EDA) - more granular and more directly useful for a
   dashboard.
4. **Prophet or ARIMA** - classical time series models handle irregular
   reporting gaps more gracefully than the lag-feature approach used here;
   worth comparing.
5. **Spike classification** - flip the problem: instead of predicting the
   exact price, predict "will tomorrow's price jump >X% from today" - a
   simpler, often more actionable target for procurement/farmer alerts.
6. **Feed this into your dashboard** - the LightGBM predictions could
   become a new "Predicted Price (Next Report)" card alongside your
   existing Price Spike Alert and Market Recommendation Engine.
