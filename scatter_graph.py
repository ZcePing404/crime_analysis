import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
clean_df = pd.read_csv('./dataset/clean_dataset.csv')
cols = ['PctPersDenseHous', 'PctKids2Par', 'PctIlleg', 'HousVacant', 'racepctblack']

# Plot scatter plots with regression lines
for col in cols:
    plt.figure(figsize=(6,4))
    sns.regplot(
        x=clean_df[col], 
        y=clean_df['ViolentCrimesPerPop'], 
        order=2,
        scatter_kws={'alpha':0.2, 's':10},  # transparency & size for better clarity
        line_kws={'color':'skyblue', 'linewidth':1}
    )
    plt.title(f"Relationship between {col} and ViolentCrimesPerPop")
    plt.xlabel(col)
    plt.ylabel("ViolentCrimesPerPop")
    plt.tight_layout()
    plt.savefig(f"graph/scatter/scatter_{col}.png", dpi=300)
    plt.close()
