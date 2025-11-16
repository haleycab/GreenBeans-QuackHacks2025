import pandas as pd
import os

''' Merge all the data. Language data from model, company ESG and emissions data'''

# 1) Build text_metrics_2024.csv
RESULTS_PATH = "results.csv"
OUT_METRICS_PATH = "data/text_metrics_2024.csv"

# Load results
results = pd.read_csv(RESULTS_PATH)

# Extract ticker from model output name (e.g. "xel_2024_chunks" -> "XEL")
results["ticker"] = (
    results["name"]
    .str.split("_", n=1).str[0]
    .str.upper()
)

# Keep columns of interest
metrics_cols = [
    "ticker",
    "relate",
    "spec",
    "senti",
    "commit",
    "metrics",
    "strategy",
    "governance",
    "risk",
]

metrics_df = results[metrics_cols].copy()

# Drop potential duplicates (if multiple files per ticker)
metrics_df = metrics_df.drop_duplicates(subset=["ticker"])

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_METRICS_PATH), exist_ok=True)

metrics_df.to_csv(OUT_METRICS_PATH, index=False)
print(f"Saved text metrics → {OUT_METRICS_PATH}")


# 2) Merge with ESG scores
TEXT_METRICS_PATH = OUT_METRICS_PATH
ESG_SCORES_PATH = "data/esg_scores.csv"
OUT_MERGED_ESG_TEXT_PATH = "data/esg_with_text_metrics_2024.csv"

# Load text metrics
metrics = pd.read_csv(TEXT_METRICS_PATH)
metrics["ticker"] = metrics["ticker"].str.upper()

# Load ESG scores
esg = pd.read_csv(ESG_SCORES_PATH)

# Normalize ESG ticker column
if "ticker" in esg.columns:
    esg["ticker"] = esg["ticker"].astype(str).str.upper()
elif "Symbol" in esg.columns:
    esg["ticker"] = esg["Symbol"].astype(str).str.upper()
else:
    raise ValueError("Could not find a 'ticker' or 'Symbol' column in esg_scores.csv")

# Merge ESG + text metrics
merged_esg_text = esg.merge(metrics, on="ticker", how="inner")

print(f"\nESG + text metrics merged shape: {merged_esg_text.shape}")
print("\nESG + text metrics example rows:")
print(merged_esg_text.head())

# Save intermediate
os.makedirs(os.path.dirname(OUT_MERGED_ESG_TEXT_PATH), exist_ok=True)
merged_esg_text.to_csv(OUT_MERGED_ESG_TEXT_PATH, index=False)
print(f"\nSaved ESG + text metrics → {OUT_MERGED_ESG_TEXT_PATH}")


# 3) Merge in emissions totals
EMISSIONS_PATH = "data/emissions_totals.csv"
OUT_MAIN_DATASET_PATH = "data/main_dataset_2024.csv"

# Load emissions
emissions = pd.read_csv(EMISSIONS_PATH)

# Normalize ticker and drop duplicates
emissions["ticker"] = emissions["ticker"].astype(str).str.upper()
emissions = emissions.drop_duplicates(subset=["ticker"])

# We only really need ticker + all_total_emissions (name is redundant with ESG)
emissions_subset = emissions[["ticker", "all_total_emissions"]].copy()

# Merge all together
main_df = merged_esg_text.merge(emissions_subset, on="ticker", how="left")

print(f"\nFinal main dataset shape (ESG + text + emissions): {main_df.shape}")
print("\nMain dataset example rows:")
print(main_df.head())

# Save main dataset
os.makedirs(os.path.dirname(OUT_MAIN_DATASET_PATH), exist_ok=True)
main_df.to_csv(OUT_MAIN_DATASET_PATH, index=False)
print(f"\nSaved main dataset → {OUT_MAIN_DATASET_PATH}")
