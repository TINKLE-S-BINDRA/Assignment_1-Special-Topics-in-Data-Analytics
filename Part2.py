import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

CSV_PATH = r"C:\Desktop\all_stocks_5yr.csv"

t0 = time.perf_counter()
df = pl.read_csv(CSV_PATH).sort(["name", "date"])
t1 = time.perf_counter()
df = df.with_columns([
    pl.col("close").rolling_mean(14).over("name").alias("SMA_14"),
    pl.col("close").rolling_mean(50).over("name").alias("SMA_50"),
])
diff = pl.col("close").diff().over("name")
df = df.with_columns(
    (100 - 100 / (1 +
     diff.clip(0, None).rolling_mean(14).over("name") /(diff.clip(None, 0).abs().rolling_mean(14).over("name") + 1e-9)
    )).alias("RSI_14")
)
df = df.drop_nulls()
t2 = time.perf_counter()
df = df.to_pandas()
df["target"] = df.groupby("name")["close"].shift(-1)
df = df.dropna()
df["rank"] = df.groupby("name").cumcount()
df["max_rank"] = df.groupby("name")["rank"].transform("max")
train = df[df["rank"] <= 0.8 * df["max_rank"]]
test  = df[df["rank"] >  0.8 * df["max_rank"]].copy()
features = ["open","high","low","close","volume","SMA_14","SMA_50","RSI_14"]
X_train, y_train = train[features], train["target"]
X_test,  y_test  = test[features],  test["target"]
t3 = time.perf_counter()
lr = LinearRegression().fit(X_train, y_train)
t4 = time.perf_counter()
pred_lr = lr.predict(X_test)
t5 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train)
t6 = time.perf_counter()
pred_rf = rf.predict(X_test)

test["pred_lr"] = pred_lr
test["pred_rf"] = pred_rf
test.to_csv("predictions.csv", index=False)
print("Polars read time:", time.perf_counter() - t0)
print("Indicators time:", time.perf_counter() - t1)
print("To pandas time:", time.perf_counter() - t2)
print("LinearRegression train time:", time.perf_counter() - t3)
print("LinearRegression test time:", time.perf_counter() - t4)
print("LinearRegression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("RandomForest train time:", time.perf_counter() - t5)
print("RandomForest test time:", time.perf_counter() - t6)
print("RandomForest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))