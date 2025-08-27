import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
column = 'PctIlleg'

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(clean_df[column], bins=50, kde=True, color='skyblue', edgecolor='black')

plt.title(f'Distribution of {column}', fontsize=14)
plt.xlabel(column)
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("graph/distribution/PctIlleg.png", dpi=300, bbox_inches='tight')