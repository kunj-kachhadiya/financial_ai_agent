import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load cleaned data
df = pd.read_csv(
    "data/raw/AAPL_cleaned.csv",
    index_col="Date",
    parse_dates=True
)

# Use Close price only
close_prices = df[["Close"]].values

# Scale data (VERY IMPORTANT)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Save scaler
joblib.dump(scaler, "data/raw/scaler.save")

# Create sequences
WINDOW_SIZE = 10

X, y = [], []

for i in range(len(scaled_prices) - WINDOW_SIZE):
    X.append(scaled_prices[i:i + WINDOW_SIZE])
    y.append(scaled_prices[i + WINDOW_SIZE])

X = np.array(X)
y = np.array(y)

print("✅ LSTM dataset ready")
print("X shape:", X.shape)  # (samples, timesteps, features)
print("y shape:", y.shape)

# Save arrays
np.save("data/raw/X_lstm.npy", X)
np.save("data/raw/y_lstm.npy", y)
