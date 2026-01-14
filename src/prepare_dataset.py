import pandas as pd

# Load cleaned data
df = pd.read_csv(
    "data/raw/AAPL_cleaned.csv",
    index_col="Date",
    parse_dates=True
)

# Use Close price only for baseline
prices = df["Close"]

WINDOW_SIZE = 10  # last 10 days → predict next day

X, y = [], []

for i in range(len(prices) - WINDOW_SIZE):
    X.append(prices.iloc[i:i + WINDOW_SIZE].values)
    y.append(prices.iloc[i + WINDOW_SIZE])

# Convert to DataFrame
X = pd.DataFrame(X)
y = pd.Series(y)

# Save datasets
X.to_csv("data/raw/X.csv", index=False)
y.to_csv("data/raw/y.csv", index=False)

print("✅ Supervised dataset created")
print("X shape:", X.shape)
print("y shape:", y.shape)
