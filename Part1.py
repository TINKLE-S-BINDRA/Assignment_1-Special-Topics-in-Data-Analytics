import polars as pl
import time
import os

CSV_PATH = r"C:\Desktop\all_stocks_5yr.csv"

start = time.perf_counter()
df_csv = pl.read_csv(CSV_PATH)
csv_read_time = time.perf_counter() - start

csv_size_mb = os.path.getsize(CSV_PATH) /(1024 * 1024)
PARQUET_PATH = "all_stocks_5yr.parquet"

df_csv.write_parquet(
    PARQUET_PATH,
    compression="zstd"
)
parquet_size_mb = os.path.getsize(PARQUET_PATH) /(1024 * 1024)

start = time.perf_counter()
df_parquet = pl.read_parquet(PARQUET_PATH)
parquet_read_time = time.perf_counter() - start

start = time.perf_counter()
df_cols = pl.read_parquet(PARQUET_PATH,columns=["date", "name", "close"] 
)
column_read_time = time.perf_counter() - start

def simulate_scale(df, scale_factor):
    return pl.concat([df] * scale_factor)

df_10x = simulate_scale(df_csv, 10)
df_10x.write_parquet("all_stocks_5yr_10x.parquet", compression="zstd")

start = time.perf_counter()
pl.read_parquet("all_stocks_5yr_10x.parquet")
time_10x = time.perf_counter() - start

size_10x = os.path.getsize("all_stocks_5yr_10x.parquet") / (1024 * 1024)

df_100x = simulate_scale(df_csv, 100)
df_100x.write_parquet("all_stocks_5yr_100x.parquet", compression="zstd")

start = time.perf_counter()
pl.read_parquet("all_stocks_5yr_100x.parquet")
time_100x = time.perf_counter() - start

size_100x = os.path.getsize("all_stocks_5yr_100x.parquet") / (1024 * 1024)

print(f"CSV: {csv_size_mb:.2f} MB in {csv_read_time:.2f} sec")
print(f"Parquet 1x : {parquet_size_mb:.2f} MB in {parquet_read_time:.2f} sec")
print(f"Parquet 10x: {size_10x:.2f} MB in {time_10x:.2f} sec")
print(f"Parquet100x: {size_100x:.2f} MB in {time_100x:.2f} sec")