import pandas as pd
import re

# === Load CSV ===
df = pd.read_csv(
    r"C:\Users\trand27\Python Projects\Bertopic Test\glassdoor_jobs_cleaned_(2023)_processed.csv",
    encoding="latin-1",
    engine="python"
)

# === 1) Reclassify the 'Size' column based on specific matching strings ===
def classify_size(value):
    if not isinstance(value, str) or value.strip() == "" or value.strip() == "N/A":
        return "Unknown"

    v = value.strip()

    # 1 to 200 group
    if v in ["1 to 50 Employees", "51 to 200 Employees"]:
        return "1 to 200 Employees"

    # 200 to 1000 group
    if v in ["201 to 500 Employees", "501 to 1000 Employees"]:
        return "200 to 1000 Employees"

    # 1000+ group
    if v in ["1001 to 5000 Employees", "5001 to 10000 Employees", "10000+ Employees"]:
        return "1000+ Employees"

    # If nothing matches
    return "Unknown"

df["Size"] = df["Size"].apply(classify_size)

# === 2) Insert MID column (median of lower + upper) ===
df['lower'] = pd.to_numeric(df['lower'], errors='coerce')
df['upper'] = pd.to_numeric(df['upper'], errors='coerce')
df['mid'] = (df['lower'] + df['upper']) / 2

# Place mid between lower and upper
cols = list(df.columns)
insert_at = cols.index("upper")
cols.insert(insert_at, cols.pop(cols.index("mid")))
df = df[cols]

# === Save CSV ===
df.to_csv("updated_file.csv", index=False)

print("Renaming complete — new file saved as updated_file.csv")
