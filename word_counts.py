import pandas as pd
import re
from collections import Counter

# Load dataset
df = pd.read_csv("data/local_spectest.csv")

# Ensure text is string
df["text"] = df["text"].astype(str)

# ----------------------------------
# Your terms (case-insensitive)
# ----------------------------------
GREEN_TERMS = [
    "science-based target", "sbti",
    "scope 1", "scope 2", "scope 3",
    "absolute emissions", "net zero",
    "1.5°", "1.5c", "paris aligned", "paris-aligned",
    "independent assurance", "limited assurance", "reasonable assurance",
    "internal carbon price", "carbon pricing",
    "tcfd", "scenario analysis",
]

RED_TERMS = [
    "aims to", "seeks to", "intends to", "aspire to",
    "where feasible", "where appropriate", "subject to",
    "forward-looking statements",
    "may ", "might ", "could ",  # note the space matters
    "emissions intensity", "offsets", "carbon credits",
]

# ----------------------------------
# Precompile regex patterns
# Each term → a case-insensitive regex
# ----------------------------------
GREEN_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE) for t in GREEN_TERMS]
RED_PATTERNS   = [re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE) for t in RED_TERMS]


# ----------------------------------
# Count matches inside a single chunk
# ----------------------------------
def count_matches(text, patterns):
    """
    Count total number of matches of a list of regex patterns in a single text chunk.
    """
    total = 0
    for p in patterns:
        total += len(p.findall(text))
    return total


# ----------------------------------
# Apply to each row
# ----------------------------------
df["green_count"] = df["text"].apply(lambda t: count_matches(t, GREEN_PATTERNS))
df["red_count"]   = df["text"].apply(lambda t: count_matches(t, RED_PATTERNS))
df["word_count"]  = df["text"].apply(lambda t: len(t.split()))


# ----------------------------------
# Aggregate scores at company-level
# ----------------------------------
company_stats = (
    df.groupby("ticker")
      .agg(
          total_chunks     = ("text", "count"),
          total_words      = ("word_count", "sum"),
          green_hits       = ("green_count", "sum"),
          red_hits         = ("red_count", "sum"),
      )
      .reset_index()
)

# Ratios
company_stats["green_per_1000w"] = company_stats["green_hits"] / (company_stats["total_words"] / 1000 + 1e-9)
company_stats["red_per_1000w"]   = company_stats["red_hits"] / (company_stats["total_words"] / 1000 + 1e-9)
company_stats["green_red_ratio"] = company_stats["green_hits"] / (company_stats["red_hits"] + 1e-9)


# ----------------------------------
# Sort for quick inspection
# ----------------------------------
company_stats_sorted = company_stats.sort_values("green_red_ratio", ascending=False)

print(company_stats_sorted)
