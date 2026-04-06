# Demand Forecasting — Store Sales

**Time Series Forecasting using ARIMA, Facebook Prophet, and XGBoost**  
Dataset: [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## Project Overview

This project builds an end-to-end demand forecasting pipeline for retail store sales. It combines three complementary forecasting approaches into a weighted ensemble and serves predictions through an interactive Streamlit dashboard.

**Models used:**
- **ARIMA (2,1,2)** — Classical statistical model for capturing linear trends and autocorrelation
- **Facebook Prophet** — Handles seasonality, holidays, and external regressors (oil price, promotions)
- **XGBoost** — Gradient boosted trees with lag features, calendar features, and rolling statistics
- **Weighted Ensemble** — Combines all three (weights: 0.20 / 0.35 / 0.45)

**Evaluation metrics:** RMSE, MAPE

---

## Project Structure

```
demand_forecasting/
├── forecasting.py      # Main pipeline: load → train → evaluate → forecast
├── dashboard.py        # Streamlit interactive dashboard
├── requirements.txt    # Dependencies
├── data/               # Place Kaggle CSVs here (or use synthetic data)
│   ├── train.csv
│   ├── stores.csv
│   └── oil.csv
└── outputs/            # Auto-created: predictions, metrics, plots
    ├── test_predictions.csv
    ├── future_forecast.csv
    ├── metrics.csv
    └── forecast_results.png
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download Kaggle dataset
```bash
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d data/
```
> If `data/train.csv` is not found, the pipeline auto-generates realistic synthetic data so you can run it immediately without Kaggle credentials.

### 3. Run the forecasting pipeline
```bash
python forecasting.py
```
This will train all models, print evaluation metrics, save CSVs to `outputs/`, and display a matplotlib summary plot.

### 4. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## Key Features

- **No Kaggle account required** — synthetic fallback data included
- **Modular pipeline** — each model is independently swappable
- **Iterative XGBoost forecasting** — generates future predictions using lagged outputs
- **Interactive dashboard** — toggle models, date range, confidence bands; download CSVs
- **Residual analysis** — visualise model errors over time and distribution

---

## Results (Synthetic Data Baseline)

| Model    | RMSE  | MAPE  |
|----------|-------|-------|
| ARIMA    | ~25.0 | ~9.5% |
| Prophet  | ~17.0 | ~6.5% |
| XGBoost  | ~13.0 | ~5.0% |
| Ensemble | ~12.5 | ~4.8% |

*Actual results vary with the Kaggle dataset.*

---

## Tech Stack

`Python` · `Pandas` · `Numpy` · `Scikit-learn` · `XGBoost` · `Facebook Prophet` · `Statsmodels` · `Streamlit` · `Plotly` · `Matplotlib`

---

## Author

**Palagiri Shashank Reddy**  
[LinkedIn](#) · [GitHub](#)
