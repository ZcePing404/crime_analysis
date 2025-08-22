import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_highly_correlated_features(df, threshold, target_col='cid',):
    # Calculate correlation
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_col].abs().sort_values()

    # Filter columns with high correlation (including the target itself)
    high_corr_features = target_corr[target_corr >= threshold].drop(target_col).index.tolist()

    print("Selected features:", high_corr_features)

    # Return a new DataFrame with only the selected features + target
    filtered_df = df[high_corr_features + [target_col]]

    return filtered_df





if __name__ == "__main__":
    # Load dataset
    clean_df = pd.read_csv('./dataset/clean_dataset.csv')

    # Compute correlation matrix (all columns)
    correlation_matrix = clean_df.corr()

    # Plot heatmap
    plt.figure(figsize=(15, 12))  # adjust size for readability
    sns.set_style("whitegrid")

    heatmap = sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",   # color scheme (blue-red)
        annot=False,       # set True if you want the values written
        fmt=".2f",         # number format
        linewidths=0.5,    # grid lines
        cbar=True,  # show color bar
        vmin=-1,   # force lower bound
        vmax=1     # force upper bound
    )

    plt.title("Correlation Heatmap of All Attributes", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("graph/correlation_coefficient.png", dpi=300, bbox_inches='tight')