import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
target_col = 'ViolentCrimesPerPop'

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(clean_df[target_col], bins=30, kde=True, color='skyblue', edgecolor='black')

plt.title(f'Distribution of {target_col}', fontsize=14)
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("graph/violentCrimesPerPop_original.png", dpi=300, bbox_inches='tight')  # save before plt.show()