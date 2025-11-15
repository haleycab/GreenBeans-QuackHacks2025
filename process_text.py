import pandas as pd
import textwrap
import funcs

reports = pd.read_csv("data/sustainability_reports.csv")
preprocessed_content = reports['preprocessed_content']

# Apply to each row
reports["processed_lines"] = reports["preprocessed_content"].apply(funcs.split_and_quote)

# 1) Save FULL DATASET
full_output_path = "data/processed_reports_text.txt"

with open(full_output_path, "w", encoding="utf-8") as f:
    for lines in reports["processed_lines"]:
        for line in lines:
            f.write(line + "\n")
        f.write("\n")   # blank line between companies

print("Saved full dataset to:", full_output_path)


# 2) Save SAMPLE OF 5 COMPANIES
# Choose first 5 — OR use: reports.sample(5, random_state=42)
sample_df = reports.iloc[:5]

sample_output_path = "data/processed_reports_sample_5.txt"

with open(sample_output_path, "w", encoding="utf-8") as f:
    for lines in sample_df["processed_lines"]:
        for line in lines:
            f.write(line + "\n")
        f.write("\n")   # blank line between companies

print("Saved sample dataset to:", sample_output_path)
