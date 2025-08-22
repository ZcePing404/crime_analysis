import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
target_col = 'cid'

# Plot binary distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=clean_df[target_col], palette="pastel", edgecolor=None)

# Add labels
plt.title('Distribution of Censoring Indicator', fontsize=14)
plt.xlabel(target_col)
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Annotate counts on bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("graph/censoringIndicator.png", dpi=300, bbox_inches='tight')
