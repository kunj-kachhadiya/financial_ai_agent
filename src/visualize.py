import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv(
    "data/raw/AAPL_cleaned.csv",
    index_col="Date",
    parse_dates=True
)

# Calculate moving averages
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# Plot
plt.figure(figsize=(14, 7))

plt.plot(df.index, df["Close"], label="Close Price", linewidth=1.5)
plt.plot(df.index, df["MA20"], label="20-Day MA", linestyle="--")
plt.plot(df.index, df["MA50"], label="50-Day MA", linestyle="--")

plt.title("AAPL Price Trend with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.show()
