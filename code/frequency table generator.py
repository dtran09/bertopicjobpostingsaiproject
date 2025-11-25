import pandas as pd
import numpy as np

# === Helper: function to build frequency table ===
def freq_table(series):
    total = len(series)
    counts = series.value_counts(dropna=False)
    percents = counts / total * 100
    return pd.DataFrame({
        "count": counts,
        "percentage": percents.round(2)
    })

# === Custom categorization for SIZE ===
size_map = {
    '1 to 50 Employees': '1-200',
    '51 to 200 Employees': '1-200',
    '201 to 500 Employees': '200-1000',
    '501 to 1000 Employees': '200-1000',
    '1001 to 5000 Employees': '1000+',
    '5001 to 10000 Employees': '1000+',
    '10000+ Employees': '1000+',
    'N/A': 'Unknown'
}

# Replace string-based rating mapping with numeric grouping helper
def compute_rating_group(series):
	"""
	Convert various rating representations to groups '1'..'5' or 'Unknown'.
	Floors numeric ratings (e.g. 4.9 -> 4), clamps >=5 to '5', and treats non-numeric as 'Unknown'.
	"""
	nums = pd.to_numeric(series, errors='coerce')
	def to_group(v):
		if pd.isna(v):
			return 'Unknown'
		group = int(np.floor(v))
		if group < 1:
			return 'Unknown'
		if group >= 5:
			return '5'
		return str(group)
	return nums.map(to_group)

# === Custom categorization for RATING ===
rating_map = {
    '1': '1',
    '2.2': '2', '2.3': '2', '2.4': '2', '2.5': '2', '2.6': '2', '2.7': '2', '2.8': '2', '2.9': '2',
    '3': '3', '3.1': '3', '3.2': '3', '3.3': '3', '3.4': '3', '3.5': '3', '3.6': '3', '3.7': '3', '3.8': '3', '3.9': '3',
    '4': '4', '4.1': '4', '4.2': '4', '4.3': '4', '4.4': '4', '4.5': '4', '4.6': '4', '4.7': '4', '4.8': '4', '4.9': '4',
    '5': '5',
    'N/A': 'Unknown'
}

def load_csv_with_fallback(path, encodings=('utf-8', 'cp1252', 'latin-1')):
	"""
	Try reading a CSV using several encodings. If the C engine fails for parsing,
	fallback to the python engine and skip bad lines.
	"""
	for enc in encodings:
		try:
			return pd.read_csv(path, encoding=enc)
		except UnicodeDecodeError:
			# encoding failed, try next
			continue
		except Exception:
			# try a more permissive parser as a last resort for this encoding
			try:
				return pd.read_csv(path, encoding=enc, engine='python', on_bad_lines='skip')
			except Exception:
				continue
	raise UnicodeDecodeError(f"Unable to read '{path}' with encodings {encodings}")

df = load_csv_with_fallback("C:/Users/trand27/Python Projects/Bertopic Test/glassdoor_jobs_cleaned_(2023)_processed.csv")

def find_column(df, candidates):
	"""
	Find the first matching column in df.columns for any name in candidates.
	Matches are case-insensitive and allow partial matches.
	"""
	lower_to_orig = {c.lower(): c for c in df.columns}
	# exact (case-insensitive) match
	for cand in candidates:
		if cand.lower() in lower_to_orig:
			return lower_to_orig[cand.lower()]
	# partial match
	for col in df.columns:
		lc = col.lower()
		for cand in candidates:
			if cand.lower() in lc or lc in cand.lower():
				return col
	return None

# Resolve columns (try common alternatives)
size_col = find_column(df, ['size', 'company size', 'company_size', 'size of company', 'company_size.1'])
rating_col = find_column(df, ['rating', 'ratings', 'company rating', 'rating_out_of_5'])
state_col = find_column(df, ['state', 'location', 'job_state', 'state_province'])
type_col  = find_column(df, ['type of company', 'company type', 'type'])
sector_col = find_column(df, ['sector', 'industry', 'industry_sector'])

# Create grouped columns safely (fall back to 'Unknown' when column missing)
if size_col:
	df['size_grouped'] = df[size_col].fillna('N/A').map(size_map).fillna('Unknown')
else:
	df['size_grouped'] = 'Unknown'

if rating_col:
	# use numeric grouping to ensure '1' and '5' are captured correctly
	df['rating_grouped'] = compute_rating_group(df[rating_col].fillna('N/A'))
else:
	df['rating_grouped'] = 'Unknown'

# For frequency tables use resolved columns or a default 'Unknown' series
state_series = df[state_col].fillna('Unknown') if state_col else pd.Series(['Unknown'] * len(df))
type_series  = df[type_col].fillna('Unknown')  if type_col  else pd.Series(['Unknown'] * len(df))
sector_series= df[sector_col].fillna('Unknown')if sector_col else pd.Series(['Unknown'] * len(df))

# === Frequency tables ===
table_rating = freq_table(df['rating_grouped'])
# ensure rating table always has rows for 1..5 and Unknown
table_rating = table_rating.reindex(['1','2','3','4','5','Unknown'], fill_value=0)

table_size = freq_table(df['size_grouped'])
table_state = freq_table(state_series)
table_type = freq_table(type_series)
table_sector = freq_table(sector_series)

# --- New: combine all frequency tables into a single long table ---
def freq_to_long(freq_df, name):
    """
    Convert a freq_table DataFrame (index = category, columns=['count','percentage'])
    into a long DataFrame with columns: dimension, category, count, percentage.
    This is robust to different index names produced by reset_index().
    """
    tmp = freq_df.reset_index()
    # find the category column as the first column that's not 'count' or 'percentage'
    non_value_cols = [c for c in tmp.columns if c not in ('count', 'percentage')]
    if not non_value_cols:
        # nothing to use as category — create a placeholder
        tmp['category'] = 'Unknown'
    else:
        cat_col = non_value_cols[0]
        if cat_col != 'category':
            tmp = tmp.rename(columns={cat_col: 'category'})
    tmp['dimension'] = name
    # ensure expected columns exist
    if 'count' not in tmp.columns:
        tmp['count'] = 0
    if 'percentage' not in tmp.columns:
        tmp['percentage'] = 0.0
    return tmp[['dimension', 'category', 'count', 'percentage']]

combined = pd.concat([
    freq_to_long(table_rating, 'rating'),
    freq_to_long(table_size, 'size'),
    freq_to_long(table_state, 'state'),
    freq_to_long(table_type, 'type_of_company'),
    freq_to_long(table_sector, 'sector')
], ignore_index=True)

# === Export to Excel to keep formatting permanent ===
with pd.ExcelWriter("frequency_tables.xlsx") as writer:
    table_rating.to_excel(writer, sheet_name="rating")
    table_size.to_excel(writer, sheet_name="size")
    table_state.to_excel(writer, sheet_name="state")
    table_type.to_excel(writer, sheet_name="type_of_company")
    table_sector.to_excel(writer, sheet_name="sector")
    # combined single-sheet (long format)
    combined.to_excel(writer, sheet_name="combined", index=False)

print("Frequency tables generated and exported to frequency_tables.xlsx")
