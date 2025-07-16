import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from preprocessing import preprocessing

# Fetch dataset
communities_dataset = fetch_ucirepo(id=183)
df = pd.DataFrame(communities_dataset.data.original)

# preprocessing
clean_df = preprocessing(df)

# Export cleaned dataset
clean_df.to_csv('./dataset/clean_dataset.csv', index=True)

# Compute correlation matrix
correlation_matrix = clean_df.corr()

# Filter the correlation which above +-0.45
target_col = 'ViolentCrimesPerPop'
strong_corr = correlation_matrix[target_col][correlation_matrix[target_col].abs() > 0.45]
strong_corr_features = strong_corr.index.tolist()

# Include only strong correlated features
filtered_corr_matrix = clean_df[strong_corr_features].corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(filtered_corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Strongly Correlated Features")
plt.tight_layout()
plt.show()

