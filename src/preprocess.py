import pandas as pd

# Load raw data
df = pd.read_csv(
    "data/raw/AAPL.csv",
    index_col=0,
    parse_dates=True
)

# Name the index
df.index.name = "Date"

# Columns that must be numeric
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

# Convert columns to numeric
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with invalid data
df = df.dropna()

# Calculate daily return
df["Daily_Return"] = df["Close"].pct_change()

# Drop first NaN from pct_change
df = df.dropna()

# Save cleaned data
df.to_csv("data/raw/AAPL_cleaned.csv")

print("✅ Preprocessing completed successfully")
print(df.head())
print("\nFinal data types:\n", df.dtypes)

# import pandas as pd

# df = pd.read_csv("data/raw/AAPL.csv", index_col=0)

# print(df.head())
# print("\nDATA TYPES:\n")
# print(df.dtypes)