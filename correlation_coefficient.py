import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Export cleaned dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
target_col = 'ViolentCrimesPerPop'

# Calculate correlation with target (excluding the target itself)
correlation_matrix = clean_df.corr()[target_col].drop(target_col)

filtered_correlation = correlation_matrix[correlation_matrix.abs() > 0.25]
# Convert to DataFrame and sort
correlation_df = filtered_correlation.sort_values()
print(correlation_df.shape[0])

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