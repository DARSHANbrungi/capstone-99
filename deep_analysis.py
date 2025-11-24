import pandas as pd
import warnings
import os

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_deep_analysis(filepath):
    """
    Performs a deep exploratory data analysis on the provided dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print("--- DATASET LOADED SUCCESSFULLY ---\n")
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        print("Please make sure 'combined_dataset.csv' is in the 'iot_anomaly_detection/data/raw/' directory.")
        return

    # --- 1. Basic Information ---
    print("--- 1. Basic Dataset Information ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nData Types:")
    print(df.dtypes)
    print("-" * 50, "\n")

    # --- 2. Missing Values ---
    print("--- 2. Missing Values Analysis ---")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent})
    print(missing_df[missing_df['Missing Count'] > 0])
    if missing_df['Missing Count'].sum() == 0:
        print("No missing values found. Excellent!")
    print("-" * 50, "\n")

    # --- 3. Constant & Quasi-Constant Columns ---
    print("--- 3. Constant Value Column Analysis ---")
    unique_counts = df.nunique()
    constant_cols = unique_counts[unique_counts == 1].index.tolist()
    if constant_cols:
        print(f"Found {len(constant_cols)} constant columns (only 1 unique value):")
        for col in constant_cols:
            print(f" - {col} (Value: {df[col].iloc[0]})")
    else:
        print("No constant columns found.")
    print("-" * 50, "\n")

    # --- 4. Cardinality Analysis ---
    print("--- 4. Cardinality (Unique Values Count) ---")
    print("Number of unique values per column:")
    print(unique_counts.sort_values(ascending=False))
    print("-" * 50, "\n")

    # --- 5. Target Variable Distribution ---
    print("--- 5. Target Variable Distribution ('Communication_Issue_Type') ---")
    if 'Communication_Issue_Type' in df.columns:
        print(df['Communication_Issue_Type'].value_counts())
    else:
        print("Target column 'Communication_Issue_Type' not found.")
    print("-" * 50, "\n")

    # --- 6. Numerical Feature Analysis ---
    print("--- 6. Numerical Feature Statistical Summary ---")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.describe())
    print("-" * 50, "\n")

    # --- 7. Categorical Feature Analysis ---
    print("--- 7. Categorical Feature Value Counts ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n--- Analyzing Column: '{col}' ---")
        print(f"Unique Values: {df[col].nunique()}")
        print("Value Counts (Top 10):")
        print(df[col].value_counts().nlargest(10))
    print("-" * 50, "\n")
    
    # --- 8. Received_Payload Structure ---
    print("--- 8. Sample of 'Received_Payload' for Parsing ---")
    if 'Received_Payload' in df.columns:
        print("First 5 entries of 'Received_Payload':")
        for i, payload in enumerate(df['Received_Payload'].head()):
            print(f"{i+1}: {payload}")
    else:
        print("'Received_Payload' column not found.")
    print("-" * 50, "\n")


if __name__ == "__main__":
    # Assuming your script is in the root, and data is in the structured path
    # Note: This script should be run from the parent directory of 'iot_anomaly_detection'
    dataset_path = os.path.join('iot_anomaly_detection', 'data', 'raw', 'combined_dataset.csv')
    run_deep_analysis(dataset_path)
