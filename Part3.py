import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.express as px

df = pd.read_csv("predictions.csv")

st.sidebar.title("Dashboard")
ticker = st.sidebar.selectbox("Select Ticker", sorted(df["name"].unique()))

df_ticker = df[df["name"] == ticker].copy()
df_ticker = df_ticker.sort_values("date") 
st.title(f"{ticker} Price Dashboard")
st.subheader("Model Performance")
rmse_lr = np.sqrt(mean_squared_error(df_ticker["target"], df_ticker["pred_lr"]))
rmse_rf = np.sqrt(mean_squared_error(df_ticker["target"], df_ticker["pred_rf"]))

col1, col2 = st.columns(2)
col1.metric("Linear Regression RMSE", f"{rmse_lr:.4f}")
col2.metric("Random Forest RMSE", f"{rmse_rf:.4f}")

st.subheader("Actual vs Predicted Close Price")

fig = px.line(
    df_ticker,x="date",y=["target", "pred_lr", "pred_rf"],labels={"value": "Close Price", "date": "Date"},title="Actual vs Predicted Close Price"
)
st.plotly_chart(fig, use_container_width=True)
st.subheader("Technical Indicators")

fig2 = px.line(df_ticker,x="date",y=["SMA_14", "SMA_50", "RSI_14"],title="SMA and RSI Indicators"
)
st.plotly_chart(fig2, use_container_width=True)
