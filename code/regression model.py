import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import statsmodels.api as sm
from scipy import stats

# --- File Paths (Ensure these match your environment) ---
OUTPUT_BASE = r"c:\Users\trand27\Python Projects\Bertopic Test"
DISTANCES_FILE = os.path.join(OUTPUT_BASE, "paragraph_topic_distances.csv")
TOPIC_INFO_FILE = os.path.join(OUTPUT_BASE, "topic_info_full.csv")

# === Define the full path for the manual file load here using the Raw String (r"...") ===
MANUAL_DATA_PATH = r"C:\Users\trand27\Python Projects\Bertopic Test\updated_file.csv"

# --- Configuration ---
# The column we want to predict (Dependent Variable, Y)
TARGET_VARIABLE = "mid"

NUMERICAL_FEATURES = ['Rating', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5']
CATEGORICAL_FEATURES = ['state', 'Size', 'Type of ownership', 'Sector']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def load_data(file_path):
    """Loads a CSV file, attempting multiple encodings."""
    try:
        # Try default UTF-8
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to Latin-1
        df = pd.read_csv(file_path, encoding='latin-1')
    return df

def get_topic_names():
    """Loads and processes topic names for easier interpretation."""
    # This function uses load_data(TOPIC_INFO_FILE), which is correct.
    try:
        topic_info_df = load_data(TOPIC_INFO_FILE)
        # Create a dictionary mapping the topic ID (e.g., 0) to its representative name
        topic_map = topic_info_df.set_index('Topic')['Name'].to_dict()
        # Clean up the names for display
        cleaned_map = {f'd{k}': v.split('_')[1:] for k, v in topic_map.items() if k != -1}
        return cleaned_map
    except FileNotFoundError:
        print(f"Warning: Topic info file not found at {TOPIC_INFO_FILE}. Coefficients will use raw 'd#' names.")
        return {}

def run_regression():
    """
    Performs a multiple linear regression to predict salary based on numerical,
    categorical (one-hot encoded), and topic distance features.
    """
    print("--- Starting Regression Analysis with Selected Features ---")
    
    # 1. Load Data
    try:
        # Try to load the automatically generated file first (preferred)
        data_df = load_data(DISTANCES_FILE)
        print(f"Successfully loaded data from {DISTANCES_FILE}")
    except FileNotFoundError:
        # If the automatically generated file is missing, fall back to the manual path
        print(f"Error: Distances file not found at {DISTANCES_FILE}. Attempting to load manual file path: {MANUAL_DATA_PATH}")
        try:
            # FIX: Use the defined variable with the raw string notation
            data_df = load_data(MANUAL_DATA_PATH)
            print(f"Successfully loaded data from manual path: {MANUAL_DATA_PATH}")
        except FileNotFoundError:
             print(f"Error: Manual data file not found at {MANUAL_DATA_PATH}. Please check file path.")
             return
        except Exception as e:
            print(f"Error loading data from {MANUAL_DATA_PATH}: {e}")
            return


    # Deduplicate data to ensure one entry per job posting for the regression
    initial_rows = len(data_df)
    try:
        # NOTE: If 'document_id' is missing when loading the manual file, this will trigger the KeyError warning.
        data_df = data_df.drop_duplicates(subset=['document_id'])
        print(f"Deduplicated data from {initial_rows} rows to {len(data_df)} unique job postings.")
    except KeyError:
        print("Warning: 'document_id' column not found for deduplication. Skipping deduplication.")
    
    # Check if all required features and target variable are present
    required_cols = ALL_FEATURES + [TARGET_VARIABLE]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        print(f"Available columns: {list(data_df.columns)}")
        return

    # 2. Prepare Data (Cleaning and Encoding)
    
    # Drop rows where the target variable is missing or invalid (e.g., placeholder 0)
    data_clean = data_df.dropna(subset=[TARGET_VARIABLE] + NUMERICAL_FEATURES).copy()
    data_clean = data_clean[data_clean[TARGET_VARIABLE] > 100].copy()
    
    if data_clean.empty:
        print(f"Error: After cleaning, no data remains with valid '{TARGET_VARIABLE}' and numerical features.")
        return
        
    # Impute missing categorical data with a placeholder like 'Missing'
    for col in CATEGORICAL_FEATURES:
        data_clean[col] = data_clean[col].fillna('Missing')
        
    # --- One-Hot Encoding for Categorical Features ---
    X_processed = pd.get_dummies(data_clean[ALL_FEATURES], 
                                 columns=CATEGORICAL_FEATURES, 
                                 drop_first=True) # drop_first avoids multicollinearity
    
    # X and Y definitions
    X = X_processed
    Y = data_clean[TARGET_VARIABLE]
    
    print(f"Running regression on {len(X)} samples with {len(X.columns)} total features (after encoding).")

    # 3. Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    # Add a constant column for OLS (intercept term)
    X_train_const = sm.add_constant(X_train)

    # Fit OLS model
    ols_model = sm.OLS(Y_train, X_train_const).fit()

    print("\n--- OLS Statistical Summary (includes p-values) ---")
    print(ols_model.summary())

    # 4. Train Model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # 5. Predict and Evaluate
    Y_pred = model.predict(X_test)
    r_squared = r2_score(Y_test, Y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"R-squared (Coefficient of Determination) on Test Set: {r_squared:.4f}")
    print(f"(A measure of how well the combined features explain the variation in salary)")

    # 6. Analyze Coefficients
    coefficients = pd.Series(model.coef_, index=X.columns)
    
    # Load topic names for better reporting (only relevant for d0-d5 features)
    topic_names = get_topic_names()
    
    print("\n--- Regression Coefficients (Impact on Salary) ---")
    
    # Create a DataFrame for organized output
    results = []
    for feature, coef in coefficients.items():
        topic_name = ""
        # Handle Topic Distance Features (d0-d5)
        if feature.startswith('d') and feature[1:].isdigit():
            topic_name = " ".join(topic_names.get(feature, [f"Topic {feature[1:]} (Name Missing)"]))
            
        # Handle Encoded Categorical Features (e.g., Sector_Information Technology)
        elif any(feature.startswith(c) for c in CATEGORICAL_FEATURES):
            # This is a one-hot encoded variable relative to the dropped base category
            topic_name = f"Category: {feature.split('_')[0]} (Value: {feature.split('_', 1)[1]})"
        
        # Handle simple numerical features (e.g., Rating)
        elif feature in NUMERICAL_FEATURES:
             topic_name = f"Numerical Feature: {feature}"
             
        
        results.append({
            "Feature": feature,
            "Description": topic_name if topic_name else feature,
            "Coefficient": f"{coef:,.2f}",
            "Impact": "Positive" if coef > 0 else "Negative"
        })

    results_df = pd.DataFrame(results)
    
    # Sort results to clearly see the highest and lowest impacts
    results_df['Coefficient_Val'] = results_df['Coefficient'].str.replace(',', '', regex=False).astype(float)
    results_df = results_df.sort_values(by="Coefficient_Val", ascending=False).drop(columns=['Coefficient_Val'])
    
    print(results_df.to_string(index=False))
    
    # General Interpretation Notes
    print("\n--- Interpretation Notes ---")
    print("1. Positive Coefficient (Impact: Positive): Increase in this feature/category is associated with higher salary.")
    print("2. Negative Coefficient (Impact: Negative): Increase in this feature/category is associated with lower salary.")
    print("3. For Topic Distances (d0-d5): A negative coefficient means that *higher similarity* (lower distance) to that topic predicts a *higher salary*.")
    
    # Save the results
    results_path = os.path.join(OUTPUT_BASE, "regression_results_detailed.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved regression coefficients to: {results_path}")

if __name__ == "__main__":
    run_regression()