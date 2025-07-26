import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Export cleaned dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
target_col = 'ViolentCrimesPerPop'

# Calculate correlation with target (excluding the target itself)
correlation_matrix = clean_df.corr()[target_col].drop(target_col)

# Filter the correlation which above +-0.45
# strong_corr = correlation_matrix[target_col][correlation_matrix[target_col].abs() > 0.45]
# strong_corr_features = strong_corr.index.tolist()

# # Include only strong correlated features
# filtered_corr_matrix = clean_df[strong_corr_features].corr()

# Plot
# plt.figure(figsize=(10, 8))
# sns.heatmap(filtered_corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
# plt.title("Correlation Matrix of Strongly Correlated Features")
# plt.tight_layout()
# Unstack and reset index


# Convert to DataFrame and sort
correlation_df = correlation_matrix.sort_values()

# Plot
plt.figure(figsize=(20, 6))  # WIDE plot to fit all 128 attribute names
sns.set_style("whitegrid")
plt.bar(correlation_df.index, correlation_df.values, color='skyblue', edgecolor='skyblue')

plt.xticks(rotation=90, fontsize=8)  # Rotate attribute names for visibility
plt.ylabel('Correlation Coefficient')
plt.xlabel('Variable Name')
plt.title(f'Correlation of Each Attribute with {target_col}')
plt.axhline(0, color='gray', linestyle='--')  # horizontal line at 0
plt.tight_layout()
plt.show()