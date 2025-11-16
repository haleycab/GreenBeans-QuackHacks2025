import os
import glob
import pandas as pd

CHUNK_DIR = "data/processed_txt_2024"
OUT_CSV = "data/local_spectest.csv"

rows = []

for path in glob.glob(os.path.join(CHUNK_DIR, "*_chunks.txt")):
    base = os.path.basename(path)
    # Expecting something like TICKER_2024_chunks.txt
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    ticker = parts[0]
    year = parts[1]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if not text:
                continue  # skip empty lines if there are blank separators

            rows.append({
                "company": ticker,   # same as before, ticker as company placeholder
                "ticker": ticker,
                "year": year,
                "text": text,
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")
