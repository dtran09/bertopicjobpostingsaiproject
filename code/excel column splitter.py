import pandas as pd
import re

def split_salary_range(salary_str):
    """
    Extract lower and upper salary ranges from salary string
    Handles various formats like: $83K - $153K, $71K - $133K, $65.00 - $75.00 Per Hour
    """
    if pd.isna(salary_str):
        return None, None
    
    salary_str = str(salary_str)
    
    # Remove text descriptions and keep only numbers/ranges
    # Handle different formats
    if 'Per Hour' in salary_str:
        # Extract hourly rates and convert to annual (assuming 2080 hours/year)
        numbers = re.findall(r'\$?(\d+\.?\d*)', salary_str)
        if len(numbers) >= 2:
            lower = float(numbers[0]) * 2080
            upper = float(numbers[1]) * 2080
            return lower, upper
    else:
        # Handle annual salaries with K notation or full numbers
        numbers = re.findall(r'\$?(\d+\.?\d*)(K|k)?', salary_str)
        valid_numbers = []
        for num, unit in numbers:
            if num.replace('.', '').isdigit():
                value = float(num)
                if unit in ['K', 'k']:
                    value *= 1000
                valid_numbers.append(value)
        
        if len(valid_numbers) >= 2:
            return min(valid_numbers), max(valid_numbers)
        elif len(valid_numbers) == 1:
            return valid_numbers[0], valid_numbers[0]
    
    return None, None

def split_location(location_str):
    """
    Split location into city and state
    Handles formats like: "San Jose, CA", "Remote", "New York, NY"
    """
    if pd.isna(location_str):
        return None, None
    
    location_str = str(location_str).strip()
    lower_loc = location_str.lower()
    
    # Special cases requested:
    if lower_loc == 'remote':
        # If the posting is just "Remote", both city and state should be "Remote"
        return 'Remote', 'Remote'
    if lower_loc == 'united states':
        # If the posting is just "United States", both city and state should be 'N/A'
        return 'N/A', 'N/A'
    
    # Handle other previously-considered tokens (keep original behavior for California)
    if lower_loc == 'california':
        return location_str, None
    
    # Split by comma
    parts = [part.strip() for part in location_str.split(',')]
    
    if len(parts) >= 2:
        city = parts[0]
        state = parts[1]
        # Clean up state if it has extra text
        state = re.sub(r'[^A-Za-z\s]', '', state).strip()
        return city, state
    else:
        return location_str, None

# Add helper to robustly read CSV with multiple encoding fallbacks
def _read_csv_with_fallback(file_path):
    """
    Try reading file with a detected encoding (if chardet available)
    and several common encodings. If direct read fails, open the file
    with the encoding and errors='replace' and feed the file handle to pandas.
    """
    encodings_tried = []
    # Try to detect encoding if chardet is installed
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw = f.read(100000)  # sample first 100k bytes
        detected = chardet.detect(raw)
        if detected and detected.get('encoding'):
            encodings_tried.append(detected['encoding'])
    except Exception:
        # chardet not available or detection failed; continue with defaults
        pass

    # common encodings to try
    for e in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']:
        if e not in encodings_tried:
            encodings_tried.append(e)

    last_exc = None
    for enc in encodings_tried:
        try:
            # try direct read first
            df = pd.read_csv(file_path, encoding=enc)
            print(f"Read CSV using encoding: {enc}")
            return df
        except Exception as direct_exc:
            last_exc = direct_exc
            # fallback: open file with errors='replace' and let pandas parse the text
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    df = pd.read_csv(f)
                print(f"Read CSV using encoding (with errors='replace'): {enc}")
                return df
            except Exception as fh_exc:
                last_exc = fh_exc
                continue

    # If all attempts failed, raise the last exception with a helpful message
    raise UnicodeDecodeError("Unable to read file with tried encodings", b'', 0, 1, f"tried: {encodings_tried}") from last_exc

def process_glassdoor_data(file_path, output_path=None):
    """
    Main function to process the Glassdoor jobs data
    """
    # Read the CSV file with encoding fallbacks
    try:
        df = _read_csv_with_fallback(file_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        raise

    # Normalize textual/empty values to 'N/A' (only affect object/string columns)
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols):
        # strip whitespace for strings
        df[obj_cols] = df[obj_cols].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # replace empty/whitespace-only strings with 'N/A'
        df[obj_cols] = df[obj_cols].replace(r'^\s*$', 'N/A', regex=True)
        # replace common unknown tokens (case-insensitive) with 'N/A'
        df[obj_cols] = df[obj_cols].replace(
            to_replace=r'(?i)^\s*unknown(?:/ *non-? *-? *applicable)?\s*$',
            value='N/A',
            regex=True
        )
        # propagate pandas NaN -> 'N/A' for textual columns
        df[obj_cols] = df[obj_cols].fillna('N/A')

        # Additionally, ensure any revenue-like columns explicitly map Unknown/Non-Applicable -> 'N/A'
        revenue_cols = [c for c in obj_cols if 'revenue' in c.lower()]
        for c in revenue_cols:
            df[c] = df[c].replace(
                to_replace=r'(?i)^\s*unknown(?:/ *non-? *-? *applicable)?\s*$',
                value='N/A',
                regex=True
            ).replace(r'^\s*$', 'N/A', regex=True).fillna('N/A')

    print(f"Original dataset shape: {df.shape}")
    print("Processing salary and location columns...")

    # Ensure required columns exist
    required_cols = ['Salary Estimate', 'Location']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in the input file. Columns found: {list(df.columns)}")

    # Split salary column
    salary_splits = df['Salary Estimate'].apply(split_salary_range)
    df['lower'] = salary_splits.apply(lambda x: x[0] if x else None)
    df['upper'] = salary_splits.apply(lambda x: x[1] if x else None)

    # Split location column
    location_splits = df['Location'].apply(split_location)
    df['city'] = location_splits.apply(lambda x: x[0] if x else None)
    df['state'] = location_splits.apply(lambda x: x[1] if x else None)

    # Ensure numeric types for salary columns (coerce invalids to NaN)
    df['lower'] = pd.to_numeric(df['lower'], errors='coerce')
    df['upper'] = pd.to_numeric(df['upper'], errors='coerce')

    # Reorder columns for better readability (optional)
    cols = list(df.columns)
    salary_idx = cols.index('Salary Estimate')
    location_idx = cols.index('Location')

    # Insert new columns after the original ones
    new_cols = (cols[:salary_idx+1] + ['lower', 'upper'] +
                cols[salary_idx+1:location_idx+1] + ['city', 'state'] +
                cols[location_idx+1:])

    df = df[new_cols]

    # Save the processed file
    if output_path is None:
        output_path = file_path.replace('.csv', '_processed.csv')

    df.to_csv(output_path, index=False)

    print(f"Processed dataset shape: {df.shape}")
    print(f"Processed file saved as: {output_path}")

    # Show some statistics (NaN-safe)
    def _format_currency(v):
        if pd.isna(v):
            return 'N/A'
        try:
            return f'${float(v):,.2f}'
        except Exception:
            return 'N/A'

    # Ensure we have a 1-D Series even if df[col] returns a DataFrame (e.g. duplicate columns)
    def _as_series(col_name):
        col = df[col_name]
        if isinstance(col, pd.DataFrame):
            # collapse into a single Series (drop index levels created by stacking)
            return col.stack().reset_index(drop=True)
        return col

    lower_s = pd.to_numeric(_as_series('lower'), errors='coerce')
    upper_s = pd.to_numeric(_as_series('upper'), errors='coerce')
    city_s = _as_series('city').astype(object)
    state_s = _as_series('state').astype(object)

    lower_count = int(lower_s.notna().sum())
    upper_count = int(upper_s.notna().sum())
    avg_lower = lower_s.mean()
    avg_upper = upper_s.mean()

    print("\nSalary statistics:")
    print(f"Jobs with salary range: {lower_count}")
    print(f"Average lower salary: {_format_currency(avg_lower)}")
    print(f"Average upper salary: {_format_currency(avg_upper)}")

    print("\nLocation statistics:")
    print(f"Jobs with city info: {int(city_s.notna().sum())}")
    print(f"Jobs with state info: {int(state_s.notna().sum())}")
    print(f"Top 5 states: {state_s.value_counts().head(5).to_dict()}")

    return df

# Usage example
if __name__ == "__main__":
    # Specify your input file path
    input_file = "glassdoor_jobs cleaned (2023).csv"

    # Process the data
    processed_df = process_glassdoor_data(input_file)

    # Display first few rows to verify
    print("\nFirst 5 rows of processed data:")
    print(processed_df[['Salary Estimate', 'lower', 'upper', 'Location', 'city', 'state']].head())