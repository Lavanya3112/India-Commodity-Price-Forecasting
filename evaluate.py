"""
evaluate.py

Generates plots comparing actual vs predicted prices for a few commodities,
plus a model comparison bar chart. Run this after train.py.
"""

import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS_DIR = "/home/claude/commodity_price_forecasting/outputs"


def plot_commodity_forecast(test_df: pd.DataFrame, commodity: str, ax):
    subset = test_df[test_df["commodity"] == commodity].sort_values("arrival_date")
    ax.plot(subset["arrival_date"], subset["modal_price"], label="Actual", marker="o", markersize=3)
    ax.plot(subset["arrival_date"], subset["lgb_pred"], label="LightGBM", linestyle="--")
    ax.plot(subset["arrival_date"], subset["rf_pred"], label="Random Forest", linestyle=":")
    ax.set_title(commodity)
    ax.set_ylabel("Modal Price (Rs/quintal)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)


def main():
    test_df = pd.read_csv(f"{OUTPUTS_DIR}/test_predictions.csv", parse_dates=["arrival_date"])
    comparison_df = pd.read_csv(f"{OUTPUTS_DIR}/model_comparison.csv")

    # Pick the commodities with the most test data points, for the clearest plots
    top_commodities = (
        test_df.groupby("commodity").size().sort_values(ascending=False).head(4).index.tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, commodity in zip(axes.flatten(), top_commodities):
        plot_commodity_forecast(test_df, commodity, ax)
    fig.suptitle("Actual vs Predicted Price - Test Period", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUTS_DIR}/forecast_plots.png", dpi=150)
    print(f"Saved: {OUTPUTS_DIR}/forecast_plots.png")

    # Model comparison bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    comparison_df.plot(x="label", y=["mae", "rmse"], kind="bar", ax=ax2)
    ax2.set_title("Model Comparison (lower is better)")
    ax2.set_ylabel("Error (Rs)")
    ax2.set_xlabel("")
    plt.xticks(rotation=0)
    fig2.tight_layout()
    fig2.savefig(f"{OUTPUTS_DIR}/model_comparison.png", dpi=150)
    print(f"Saved: {OUTPUTS_DIR}/model_comparison.png")

    print("\nFinal comparison table:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
