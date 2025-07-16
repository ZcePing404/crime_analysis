import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch dataset
communities_dataset = fetch_ucirepo(id=183)

# Extract features and target
df = communities_dataset.data.original
df_clean = df.replace("?", 0)  # Replace missing values

# Drop columns with non-predictable columns
useless_columns = ["state", "county", "community", "communityname", "fold", "countyname"]
data = df_clean.drop(columns=useless_columns, errors='ignore')

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Compute correlation matrix
correlation_matrix = data.corr()

# Filter the correlation which above +-0.45
target_col = 'ViolentCrimesPerPop'
strong_corr = correlation_matrix[target_col][correlation_matrix[target_col].abs() > 0.45]
strong_corr_features = strong_corr.index.tolist()

# Include only strong correlated features
filtered_corr_matrix = data[strong_corr_features].corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(filtered_corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Strongly Correlated Features")
plt.tight_layout()
plt.show()

