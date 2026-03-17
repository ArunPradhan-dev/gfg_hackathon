import pandas as pd


df = pd.read_csv(
    "customer_behavior.csv",
    encoding="latin1",
    skiprows=1
)

print("Rows before cleaning:", len(df))

# Remove completely empty rows
df = df.dropna(how="all")

# Remove duplicate rows
df = df.drop_duplicates()

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")

# Remove corrupted rows containing HTML or links
df = df[~df.astype(str).apply(
    lambda x: x.str.contains("html|http", case=False)
).any(axis=1)]

# Remove rows where gender is missing
df = df[df["gender"].notna()]

# Optional: fill missing values
df["gender"] = df["gender"].fillna("Unknown")

print("Rows after cleaning:", len(df))

# Save cleaned dataset
df.to_csv("clean_customer_data.csv", index=False)

print("Clean dataset saved.")