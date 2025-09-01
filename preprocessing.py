import pandas as pd
import numpy as np
import feature_selection as fs
from ucimlrepo import fetch_ucirepo

def remove_unuseful(df):
    # Drop columns with non-predictable columns
    useless_columns = ["state", "county", "community", "communityname", "fold", "countyname"]
    df = df.drop(columns=useless_columns, errors='ignore')
    return df

def handle_missing(df):
    # Replace "?" with NaN
    df = df.replace("?", np.nan)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Identify columns with more than 50% missing values
    cols_to_drop = df.columns[df.isnull().mean() >= 0.5].tolist()

    # Drop those columns
    df = df.drop(columns=cols_to_drop)
    if cols_to_drop:
        print("Dropped columns (more than 50% missing):")
        for col in cols_to_drop:
            print(f"  - {col}")

    # Identify columns that still have missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    # Fill remaining missing values with mean
    df = df.fillna(df.mean())

    if cols_with_missing:
        print("\nColumns filled with mean:")
        for col in cols_with_missing:
            print(f"  - {col}")

    return df





# Fetch dataset
communities_dataset = fetch_ucirepo(id=183)
df = pd.DataFrame(communities_dataset.data.original)
original_df = df.copy()

print(f"\n\nInitial number of attributes: {df.shape[1]}")

# Remove unuseful attributes
before = df.shape[1]
df = remove_unuseful(df)
after = df.shape[1]
print(f"\nAfter remove unuseful features  : {after} attributes (Removed {before - after})")

# Handle missing values
before = df.shape[1]
df = handle_missing(df)
after = df.shape[1]
print(f"\nAfter handle missing values     : {after} attributes (Removed {before - after})")

df.to_csv('./dataset/clean_dataset.csv', index=False)


