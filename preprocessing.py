import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def remove_unuseful(df):
    # Drop columns with non-predictable columns
    useless_columns = ["state", "county", "community", "communityname", "fold", "countyname"]
    df = df.drop(columns=useless_columns, errors='ignore')
    return df

def handle_missing(df):
    # Drop columns with more than 50% missing values
    df = df.replace("?", np.nan)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.loc[:, df.isnull().mean() < 0.5]
    
    # Fill remaining missing values with mean
    df = df.fillna(df.mean())
    return df


def remove_redundant(df):
    abs_corr_matrix = df.corr().abs()  # Take absolute correlation

    # Creates the upper triangle of the correlation matrix — everything below the diagonal (including the diagonal) is turned into NaN
    upper = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool)
    )
    # Find columns with high correlation (e.g., > 0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    df = df.drop(columns=to_drop)
    return df


# Fetch dataset
communities_dataset = fetch_ucirepo(id=183)
df = pd.DataFrame(communities_dataset.data.original)

print(f"Initial number of attributes: {df.shape[1]}")

# Step 1: Remove unuseful attributes
before = df.shape[1]
df = remove_unuseful(df)
after = df.shape[1]
print(f"After remove unuseful features  : {after} attributes (Removed {before - after})")

# Step 2: Handle missing values
before = df.shape[1]
df = handle_missing(df)
after = df.shape[1]
print(f"After handle missing values     : {after} attributes (Removed {before - after})")

# Step 3: Remove redundant attributes
before = df.shape[1]
df = remove_redundant(df)
after = df.shape[1]
print(f"After remove redundant features : {after} attributes (Removed {before - after})")

print(f"\nFinal number of features      : {after}")

df.to_csv('./dataset/clean_dataset.csv')


