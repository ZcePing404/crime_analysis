import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
cols = ['PctPersDenseHous', 'PctKids2Par', 'PctIlleg', 'HousVacant', 'racepctblack', 'ViolentCrimesPerPop']

for col in cols:
    # Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(clean_df[col], bins=50, kde=True, color='skyblue', edgecolor='gray')

    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"graph/distribution/{col}.png", dpi=300, bbox_inches='tight')
    print(f"{col}.png saved.")
    plt.close()