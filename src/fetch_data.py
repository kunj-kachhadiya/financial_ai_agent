import yfinance as yf
import pandas as pd
import os

SYMBOL = "AAPL"
DATA_DIR = "data/raw"

os.makedirs(DATA_DIR, exist_ok=True)

df = yf.download(SYMBOL, period="5y", interval="1d")

df.to_csv(f"{DATA_DIR}/{SYMBOL}.csv")

print(f"Data fetched for {SYMBOL}")
print(df.head())
