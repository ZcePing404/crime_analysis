import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

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

def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    iqr_df = pd.DataFrame({
        'Q1 (25%)': Q1,
        'Q3 (75%)': Q3,
        'IQR': IQR
    })

    print(iqr_df)
    # Keep rows where all columns are within the IQR range
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]

def remove_multi_collinearity(df):
    abs_corr_matrix = df.corr().abs()  # Take absolute correlation

    # Creates the upper triangle of the correlation matrix — everything below the diagonal (including the diagonal) is turned into NaN
    upper = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool)
    )
    # Find columns with high correlation (e.g., > 0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    df = df.drop(columns=to_drop)
    return df

def plot_boxplots_before_after(df_before, df_after, features, ncols=3):
    n = len(features)
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(ncols*5, nrows*4))

    for i, col in enumerate(features):
        plt.subplot(nrows, ncols, i+1)
        sns.boxplot(data=pd.concat([df_before[col], df_after[col]], axis=1),palette='Set2')
        plt.xticks([0, 1], ['Before', 'After'])
        plt.title(col)

    plt.tight_layout()
    plt.show()


# Fetch dataset
communities_dataset = fetch_ucirepo(id=183)
df = pd.DataFrame(communities_dataset.data.original)
df_before_outlier_removal = df.copy()
features_to_plot = ['population', 'householdsize', 'racepctblack', 'racePctAsian', 'racePctHisp']

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

# Step 4: Remove remove multi-collinearity attiribute
before = df.shape[1]
df = remove_multi_collinearity(df)
after = df.shape[1]
print(f"After remove redundant features : {after} attributes (Removed {before - after})")

# Step 3: Remove outliers
before_rows = df.shape[0]
df = remove_outliers_iqr(df)
after_rows = df.shape[0]
print(f"After removing outliers         : {after_rows} rows (Removed {before_rows - after_rows})")

print(f"\nFinal number of features      : {after}")
print(df.columns.tolist())

df.to_csv('./dataset/clean_dataset_without_outlier.csv')

sns.boxplot(data=df_before_outlier_removal['population'])
plt.title('Before Removing Outliers')
plt.show()

# After removing outliers
sns.boxplot(data=df['population'])
plt.title('After Removing Outliers')
plt.show()


