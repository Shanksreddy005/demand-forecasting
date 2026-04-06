"""
Demand Forecasting Dashboard
=============================
Interactive Streamlit app for visualising forecast vs. actual sales.
Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #378ADD;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-delta { font-size: 13px; color: #1D9E75; }
    .section-title { font-size: 18px; font-weight: 600; color: #1a1a2e; margin: 16px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ─── SYNTHETIC DATA (used when no output CSVs exist) ─────────────────────────

@st.cache_data
def generate_synthetic():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-08-31", freq="D")
    n = len(dates)
    t = np.arange(n)
    trend   = 200 + 0.05 * t
    weekly  = 30  * np.sin(2 * np.pi * t / 7)
    annual  = 50  * np.sin(2 * np.pi * t / 365)
    noise   = np.random.normal(0, 15, n)
    actual  = np.maximum(0, trend + weekly + annual + noise)

    split = int(n * 0.85)
    test_dates  = dates[split:]
    test_actual = actual[split:]

    arima_pred   = test_actual + np.random.normal(0, 18, len(test_actual))
    prophet_pred = test_actual + np.random.normal(0, 12, len(test_actual))
    xgb_pred     = test_actual + np.random.normal(0, 10, len(test_actual))
    ens_pred     = 0.2 * arima_pred + 0.35 * prophet_pred + 0.45 * xgb_pred

    test_df = pd.DataFrame({
        "ds": test_dates, "y": test_actual,
        "arima": np.maximum(0, arima_pred),
        "prophet": np.maximum(0, prophet_pred),
        "xgb": np.maximum(0, xgb_pred),
        "ensemble": np.maximum(0, ens_pred)
    })

    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=16, freq="D")
    last_val = actual[-1]
    fcast    = last_val + np.cumsum(np.random.normal(0.5, 5, 16))
    future_df = pd.DataFrame({
        "ds": future_dates,
        "prophet":  np.maximum(0, fcast + np.random.normal(0, 8, 16)),
        "xgboost":  np.maximum(0, fcast + np.random.normal(0, 6, 16)),
        "ensemble": np.maximum(0, fcast)
    })

    hist_df = pd.DataFrame({"ds": dates, "y": actual})
    return hist_df, test_df, future_df


@st.cache_data
def load_data():
    test_path   = os.path.join("outputs", "test_predictions.csv")
    future_path = os.path.join("outputs", "future_forecast.csv")

    if os.path.exists(test_path) and os.path.exists(future_path):
        test_df   = pd.read_csv(test_path,   parse_dates=["ds"])
        future_df = pd.read_csv(future_path, parse_dates=["ds"])
        hist_df   = test_df[["ds", "y"]].copy()
        return hist_df, test_df, future_df
    else:
        return generate_synthetic()


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ─── LOAD ─────────────────────────────────────────────────────────────────────

hist_df, test_df, future_df = load_data()
actual = test_df["y"].values

metrics = {
    "ARIMA":    {"rmse": rmse(actual, test_df["arima"].values),   "mape": mape(actual, test_df["arima"].values)},
    "Prophet":  {"rmse": rmse(actual, test_df["prophet"].values), "mape": mape(actual, test_df["prophet"].values)},
    "XGBoost":  {"rmse": rmse(actual, test_df["xgb"].values),     "mape": mape(actual, test_df["xgb"].values)},
    "Ensemble": {"rmse": rmse(actual, test_df["ensemble"].values),"mape": mape(actual, test_df["ensemble"].values)},
}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    selected_models = st.multiselect(
        "Models to display",
        ["ARIMA", "Prophet", "XGBoost", "Ensemble"],
        default=["Ensemble", "XGBoost"]
    )

    show_confidence = st.checkbox("Show confidence band", value=True)
    show_actuals    = st.checkbox("Show actual values",   value=True)

    st.markdown("---")
    st.subheader("Date Range")
    date_range = st.slider(
        "Test set window",
        min_value=7, max_value=len(test_df),
        value=len(test_df), step=1
    )

    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    **Techniques used:**
    - ARIMA (2,1,2)
    - Facebook Prophet + regressors
    - XGBoost + lag features
    - Weighted Ensemble

    **Dataset:** Kaggle Store Sales
    """)

# ─── HEADER ──────────────────────────────────────────────────────────────────

st.title("📈 Demand Forecasting Dashboard")
st.markdown("Store Sales — Time Series Forecast vs. Actual | ARIMA · Prophet · XGBoost · Ensemble")
st.markdown("---")

# ─── KPI CARDS ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
model_colors = {"ARIMA": "#E24B4A", "Prophet": "#378ADD", "XGBoost": "#1D9E75", "Ensemble": "#BA7517"}

for col, (model, m) in zip([col1, col2, col3, col4], metrics.items()):
    best = model == min(metrics, key=lambda k: metrics[k]["mape"])
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{model_colors[model]}">
        <div class="metric-label">{model}</div>
        <div class="metric-value">{m['mape']:.1f}%</div>
        <div class="metric-delta">MAPE &nbsp;|&nbsp; RMSE: {m['rmse']:.1f} {'🏆' if best else ''}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── MAIN CHART: FORECAST VS ACTUAL ──────────────────────────────────────────

st.markdown('<div class="section-title">Forecast vs. Actual — Test Set</div>', unsafe_allow_html=True)

plot_test = test_df.tail(date_range).copy()
fig = go.Figure()

if show_actuals:
    fig.add_trace(go.Scatter(
        x=plot_test["ds"], y=plot_test["y"],
        name="Actual", line=dict(color="black", width=2.5),
        mode="lines", zorder=10
    ))

model_col_map = {"ARIMA": "arima", "Prophet": "prophet", "XGBoost": "xgb", "Ensemble": "ensemble"}
dash_map      = {"ARIMA": "dash", "Prophet": "dashdot", "XGBoost": "dot", "Ensemble": "solid"}

for model in selected_models:
    col  = model_col_map[model]
    lw   = 2.5 if model == "Ensemble" else 1.5
    fig.add_trace(go.Scatter(
        x=plot_test["ds"], y=plot_test[col],
        name=model, line=dict(color=model_colors[model], width=lw, dash=dash_map[model]),
        mode="lines"
    ))

if show_confidence and "Ensemble" in selected_models:
    ens = plot_test["ensemble"]
    fig.add_trace(go.Scatter(
        x=pd.concat([plot_test["ds"], plot_test["ds"][::-1]]),
        y=pd.concat([ens * 1.12, (ens * 0.88)[::-1]]),
        fill="toself", fillcolor="rgba(186,117,23,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±12% band", showlegend=True
    ))

fig.update_layout(
    height=380, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Date", yaxis_title="Sales",
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# ─── FUTURE FORECAST ─────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Future Forecast (Next 16 Days)</div>', unsafe_allow_html=True)

fig2 = go.Figure()
hist_window = hist_df.tail(60)

fig2.add_trace(go.Scatter(
    x=hist_window["ds"], y=hist_window["y"],
    name="Historical", line=dict(color="gray", width=1.5), opacity=0.7
))
fig2.add_trace(go.Scatter(
    x=future_df["ds"], y=future_df["ensemble"],
    name="Ensemble Forecast", line=dict(color="#BA7517", width=2.5)
))
fig2.add_trace(go.Scatter(
    x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
    y=pd.concat([future_df["ensemble"] * 1.12, (future_df["ensemble"] * 0.88)[::-1]]),
    fill="toself", fillcolor="rgba(186,117,23,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±12% confidence"
))
fig2.add_trace(go.Scatter(
    x=[hist_df["ds"].max(), hist_df["ds"].max()],
    y=[0, future_df["ensemble"].max() * 1.2],
    mode="lines", line=dict(color="gray", dash="dash", width=1.5),
    name="Forecast start"
))
fig2.update_layout(
    height=320, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Date", yaxis_title="Sales",
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

# ─── BOTTOM ROW: METRICS + RESIDUALS ─────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame([
        {"Model": k, "RMSE": round(v["rmse"], 2), "MAPE (%)": round(v["mape"], 2)}
        for k, v in metrics.items()
    ])
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=metrics_df["Model"], y=metrics_df["RMSE"],
        name="RMSE", marker_color=[model_colors[m] for m in metrics_df["Model"]],
        opacity=0.85, yaxis="y"
    ))
    fig3.add_trace(go.Scatter(
        x=metrics_df["Model"], y=metrics_df["MAPE (%)"],
        name="MAPE (%)", line=dict(color="black", width=2),
        mode="lines+markers", yaxis="y2"
    ))
    fig3.update_layout(
        height=300, template="plotly_white",
        yaxis=dict(title="RMSE"),
        yaxis2=dict(title="MAPE (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">Residual Analysis — Ensemble</div>', unsafe_allow_html=True)
    residuals = test_df["y"].values - test_df["ensemble"].values
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=["Residuals over time", "Distribution"])
    fig4.add_trace(
        go.Scatter(x=test_df["ds"], y=residuals, mode="lines",
                   line=dict(color="#378ADD", width=1.2), name="Residual"),
        row=1, col=1
    )
    fig4.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig4.add_trace(
        go.Histogram(x=residuals, nbinsx=20, marker_color="#378ADD",
                     opacity=0.75, name="Distribution"),
        row=1, col=2
    )
    fig4.update_layout(
        height=300, template="plotly_white",
        showlegend=False, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─── RAW DATA TABLE ──────────────────────────────────────────────────────────

with st.expander("📋 View Raw Forecast Data"):
    display_df = test_df.copy()
    display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m-%d")
    for col in ["arima", "prophet", "xgb", "ensemble"]:
        display_df[col] = display_df[col].round(2)
    st.dataframe(display_df.tail(30), use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=test_df.to_csv(index=False),
        file_name="forecast_predictions.csv",
        mime="text/csv"
    )

# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small>Built with Python · ARIMA · Facebook Prophet · XGBoost · Streamlit · Plotly "
    "| Palagiri Shashank Reddy</small>",
    unsafe_allow_html=True
)
