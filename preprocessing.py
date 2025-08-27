import pandas as pd
import numpy as np
import correlation_coefficient as cc
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


def remove_multi_collinearity(df, label_col='ViolentCrimesPerPop', threshold=0.85):
    df = df.copy()

    # Compute correlation matrix
    corr_matrix = df.drop(columns=[label_col]).corr()
    abs_corr_matrix = corr_matrix.abs()

    # Find all highly correlated feature pairs
    abs_corr_pairs = abs_corr_matrix.unstack().sort_values(ascending=False)
    high_corr = [(a, b) for a, b in abs_corr_pairs.index if a != b and abs_corr_pairs[(a, b)] > threshold]
    
    removed = set()
    to_remove = []

    print("\n\nHighly correlated pairs (corr > {}):\n".format(threshold))
    for a, b in high_corr:
        if a in removed or b in removed:
            continue  # Skip if already removed

        # Compare correlation with target
        a_corr = df[[a, label_col]].corr().iloc[0, 1]
        b_corr = df[[b, label_col]].corr().iloc[0, 1]
        # Check if both is same neg or pos
        if (a_corr * b_corr > 0):
            # Keep the one more related to the label
            if abs(a_corr) < abs(b_corr):
                to_remove.append(a)
                removed.add(a)
                print(f"Removing '{a}' (corr with target = {a_corr:.4f}) highly correlated with '{b}' (corr with target = {b_corr:.4f}) [corr = {abs_corr_pairs[(a, b)]:.4f}]")
            else:
                to_remove.append(b)
                removed.add(b)
                print(f"Removing '{b}' (corr with target = {b_corr:.4f}) highly correlated with '{a}' (corr with target = {a_corr:.4f}) [corr = {abs_corr_pairs[(a, b)]:.4f}]")

    # Return a new DataFrame with selected features
    cleaned_df = df.drop(columns=to_remove)
    return cleaned_df



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

# Remove multi-collinearity attiribute
before = df.shape[1]
df = remove_multi_collinearity(df)
after = df.shape[1]
print(f"\nAfter remove redundant features : {after} attributes (Removed {before - after})")

df.to_csv('./dataset/clean_dataset.csv', index=False)

# Select only highly related features
before = df.shape[1]
df = cc.get_highly_correlated_features(df, threshold=0.55)
after = df.shape[1]

print(f"\nFinal number of features        : {after}")
print(df.columns.tolist())
df.to_csv('./dataset/final_dataset.csv', index=False)


