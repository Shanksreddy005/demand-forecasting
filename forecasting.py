"""
Demand Forecasting Pipeline
===========================
Techniques: ARIMA, Facebook Prophet, XGBoost
Dataset   : Kaggle Store Sales - Time Series Forecasting
Metrics   : RMSE, MAPE
Author    : Palagiri Shashank Reddy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR   = "data"           # folder containing train.csv, stores.csv, oil.csv
OUTPUT_DIR = "outputs"
FORECAST_HORIZON = 16        # days to forecast
TEST_DAYS        = 30        # days held out for evaluation
STORE_ID         = 1         # store to analyse
FAMILY           = "GROCERY I"  # product family

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_metrics(name, y_true, y_pred):
    r = rmse(y_true, y_pred)
    m = mape(y_true, y_pred)
    print(f"  {name:<12}  RMSE: {r:8.2f}   MAPE: {m:6.2f}%")
    return {"model": name, "rmse": round(r, 2), "mape": round(m, 2)}


# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────

def load_data():
    """
    Loads Kaggle Store Sales data.
    Falls back to synthetic data if CSVs are not present so the script
    runs standalone without the dataset downloaded.
    """
    train_path = os.path.join(DATA_DIR, "train.csv")

    if os.path.exists(train_path):
        print("[INFO] Loading Kaggle Store Sales dataset...")
        train  = pd.read_csv(train_path, parse_dates=["date"])
        stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
        oil    = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])

        df = train[(train["store_nbr"] == STORE_ID) & (train["family"] == FAMILY)].copy()
        df = df.merge(stores, on="store_nbr", how="left")
        df = df.merge(oil, on="date", how="left")
        df["dcoilwtico"] = df["dcoilwtico"].interpolate(method="linear")
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "sales", "onpromotion", "dcoilwtico"]].copy()
        df.rename(columns={"sales": "y", "date": "ds"}, inplace=True)

    else:
        print("[INFO] Kaggle data not found — generating synthetic dataset...")
        df = _generate_synthetic_data()

    return df


def _generate_synthetic_data():
    """
    Generates realistic synthetic retail demand data with:
    - weekly seasonality
    - annual trend
    - promotion spikes
    - random noise
    """
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2023-08-31", freq="D")
    n = len(dates)

    t = np.arange(n)
    trend      = 200 + 0.05 * t
    weekly     = 30 * np.sin(2 * np.pi * t / 7)
    annual     = 50 * np.sin(2 * np.pi * t / 365)
    noise      = np.random.normal(0, 15, n)
    promotion  = np.where(np.random.rand(n) < 0.1, np.random.uniform(20, 80, n), 0)
    sales      = np.maximum(0, trend + weekly + annual + noise + promotion)

    oil_base   = 70 + 0.01 * t + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "ds":          dates,
        "y":           sales,
        "onpromotion": (promotion > 0).astype(int),
        "dcoilwtico":  oil_base
    })
    return df


def feature_engineering(df):
    """Adds calendar and lag features for XGBoost."""
    df = df.copy()
    df["dayofweek"]  = df["ds"].dt.dayofweek
    df["month"]      = df["ds"].dt.month
    df["dayofyear"]  = df["ds"].dt.dayofyear
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"]    = df["ds"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["rolling_mean_7"]  = df["y"].shift(1).rolling(7).mean()
    df["rolling_mean_28"] = df["y"].shift(1).rolling(28).mean()
    df["rolling_std_7"]   = df["y"].shift(1).rolling(7).std()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── MODELS ──────────────────────────────────────────────────────────────────

def run_arima(train_y, test_y):
    """Fits ARIMA(2,1,2) on training data and forecasts test period."""
    print("\n[ARIMA] Fitting model...")
    model  = ARIMA(train_y.values, order=(2, 1, 2))
    result = model.fit()
    preds  = result.forecast(steps=len(test_y))
    preds  = np.maximum(0, preds)
    return preds, result


def run_prophet(train_df, test_df):
    """
    Fits Facebook Prophet with:
    - weekly & yearly seasonality
    - oil price as external regressor
    - promotion as regressor
    """
    print("\n[Prophet] Fitting model...")

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05
    )
    m.add_regressor("onpromotion")
    m.add_regressor("dcoilwtico")

    m.fit(train_df[["ds", "y", "onpromotion", "dcoilwtico"]])

    future = test_df[["ds", "onpromotion", "dcoilwtico"]].copy()
    forecast = m.predict(future)
    preds = np.maximum(0, forecast["yhat"].values)
    return preds, m, forecast


def run_xgboost(train_fe, test_fe):
    """
    Fits XGBoost with lag + calendar features.
    Uses early stopping on a 10% validation split.
    """
    print("\n[XGBoost] Fitting model...")

    feature_cols = [
        "dayofweek", "month", "dayofyear", "weekofyear",
        "quarter", "is_weekend", "onpromotion", "dcoilwtico",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7"
    ]

    X_train = train_fe[feature_cols].values
    y_train = train_fe["y"].values
    X_test  = test_fe[feature_cols].values

    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = np.maximum(0, model.predict(X_test))
    return preds, model, feature_cols


def ensemble_forecast(preds_arima, preds_prophet, preds_xgb,
                       weights=(0.2, 0.35, 0.45)):
    """Weighted ensemble of all three models."""
    return (weights[0] * preds_arima +
            weights[1] * preds_prophet +
            weights[2] * preds_xgb)


# ─── FUTURE FORECAST ─────────────────────────────────────────────────────────

def forecast_future(df, prophet_model, xgb_model, feature_cols, horizon=FORECAST_HORIZON):
    """Generates future forecasts beyond the dataset end date."""
    last_date  = df["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future_df = pd.DataFrame({"ds": future_dates})
    future_df["onpromotion"] = 0
    future_df["dcoilwtico"]  = df["dcoilwtico"].iloc[-30:].mean()

    # Prophet future
    prophet_future = prophet_model.predict(future_df[["ds", "onpromotion", "dcoilwtico"]])
    prophet_preds  = np.maximum(0, prophet_future["yhat"].values)

    # XGBoost future (iterative, using last known values)
    extended = df[["ds", "y", "onpromotion", "dcoilwtico"]].copy()
    xgb_preds = []
    for i, row in future_df.iterrows():
        temp_row = row.copy()
        temp_row["y"] = 0
        temp_extended = pd.concat([extended, pd.DataFrame([temp_row])], ignore_index=True)
        temp_fe = feature_engineering(temp_extended)
        last_row = temp_fe.iloc[[-1]]
        available = [c for c in feature_cols if c in last_row.columns]
        pred = float(xgb_model.predict(last_row[available].values))
        pred = max(0, pred)
        xgb_preds.append(pred)
        new_row = {"ds": row["ds"], "y": pred,
                   "onpromotion": row["onpromotion"], "dcoilwtico": row["dcoilwtico"]}
        extended = pd.concat([extended, pd.DataFrame([new_row])], ignore_index=True)

    xgb_preds  = np.array(xgb_preds)
    ensemble   = 0.45 * prophet_preds + 0.55 * xgb_preds

    future_results = pd.DataFrame({
        "ds":       future_dates,
        "prophet":  prophet_preds,
        "xgboost":  xgb_preds,
        "ensemble": ensemble
    })
    return future_results


# ─── VISUALISATION ───────────────────────────────────────────────────────────

def plot_results(df, test_df, results, future_df, metrics):
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle("Demand Forecasting — Store Sales", fontsize=16, fontweight="bold", y=0.98)

    # ── Plot 1: Model Comparison ──
    ax = axes[0]
    ax.plot(test_df["ds"], test_df["y"],      color="black",  lw=2,   label="Actual",   zorder=5)
    ax.plot(test_df["ds"], results["arima"],  color="#E24B4A", lw=1.5, label="ARIMA",   linestyle="--")
    ax.plot(test_df["ds"], results["prophet"],color="#378ADD", lw=1.5, label="Prophet",  linestyle="-.")
    ax.plot(test_df["ds"], results["xgb"],    color="#1D9E75", lw=1.5, label="XGBoost", linestyle=":")
    ax.plot(test_df["ds"], results["ensemble"],color="#BA7517",lw=2,   label="Ensemble")
    ax.set_title("Model Comparison on Test Set", fontweight="bold")
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # ── Plot 2: Historical + Future Forecast ──
    ax = axes[1]
    hist_window = df.tail(90)
    ax.plot(hist_window["ds"], hist_window["y"], color="black", lw=1.5, label="Historical", alpha=0.7)
    ax.plot(future_df["ds"], future_df["ensemble"], color="#BA7517", lw=2, label=f"Forecast ({FORECAST_HORIZON}d)")
    ax.fill_between(
        future_df["ds"],
        future_df["ensemble"] * 0.88,
        future_df["ensemble"] * 1.12,
        alpha=0.25, color="#BA7517", label="±12% confidence"
    )
    ax.axvline(df["ds"].max(), color="gray", linestyle="--", alpha=0.5, label="Forecast start")
    ax.set_title(f"Historical Sales + {FORECAST_HORIZON}-Day Ensemble Forecast", fontweight="bold")
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # ── Plot 3: Metrics Bar Chart ──
    ax = axes[2]
    models = [m["model"] for m in metrics]
    rmses  = [m["rmse"]  for m in metrics]
    mapes  = [m["mape"]  for m in metrics]
    colors = ["#E24B4A", "#378ADD", "#1D9E75", "#BA7517"]
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, rmses, w, label="RMSE", color=colors, alpha=0.85)
    ax2   = ax.twinx()
    bars2 = ax2.bar(x + w/2, mapes, w, label="MAPE (%)", color=colors, alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("MAPE (%)")
    ax.set_title("Model Performance Metrics", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "forecast_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plot saved to {path}")
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  DEMAND FORECASTING PIPELINE")
    print("=" * 55)

    # 1. Load & split
    df = load_data()
    print(f"[INFO] Dataset: {len(df)} days  ({df['ds'].min().date()} → {df['ds'].max().date()})")

    split_idx = len(df) - TEST_DAYS
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    print(f"[INFO] Train: {len(train_df)} days | Test: {len(test_df)} days")

    # 2. Feature engineering for XGBoost
    full_fe  = feature_engineering(df.copy())
    train_fe = full_fe.iloc[:split_idx - 28].copy()   # account for lag window
    test_fe  = full_fe.iloc[split_idx - 28:].copy()
    test_fe  = test_fe[test_fe["ds"].isin(test_df["ds"])].copy()

    # 3. Run models
    arima_preds,  arima_model           = run_arima(train_df["y"], test_df["y"])
    prophet_preds, prophet_model, _     = run_prophet(train_df, test_df)
    xgb_preds,    xgb_model, feat_cols  = run_xgboost(train_fe, test_fe)
    ens_preds = ensemble_forecast(arima_preds, prophet_preds, xgb_preds)

    # 4. Evaluate
    print("\n" + "─" * 45)
    print("  MODEL EVALUATION METRICS")
    print("─" * 45)
    actual  = test_df["y"].values
    metrics = [
        print_metrics("ARIMA",    actual, arima_preds),
        print_metrics("Prophet",  actual, prophet_preds),
        print_metrics("XGBoost",  actual, xgb_preds),
        print_metrics("Ensemble", actual, ens_preds),
    ]

    # 5. Future forecast
    print(f"\n[INFO] Generating {FORECAST_HORIZON}-day future forecast...")
    future_df = forecast_future(df, prophet_model, xgb_model, feat_cols)

    # 6. Save results
    results_df = test_df[["ds", "y"]].copy()
    results_df["arima"]    = arima_preds
    results_df["prophet"]  = prophet_preds
    results_df["xgb"]      = xgb_preds
    results_df["ensemble"] = ens_preds
    results_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    future_df.to_csv(os.path.join(OUTPUT_DIR, "future_forecast.csv"),   index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    print(f"[INFO] Results saved to /{OUTPUT_DIR}/")

    # 7. Plot
    plot_results(
        df,
        test_df,
        {"arima": arima_preds, "prophet": prophet_preds,
         "xgb": xgb_preds, "ensemble": ens_preds},
        future_df,
        metrics
    )

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
