import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_highly_correlated_features(df, target_col='ViolentCrimesPerPop', threshold=0.6):
    # Calculate correlation
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_col].abs()

    # Filter columns with high correlation (including the target itself)
    high_corr_features = target_corr[target_corr > threshold].drop(target_col).index.tolist()

    print("Selected features:", high_corr_features)

    # Return a new DataFrame with only the selected features + target
    filtered_df = df[high_corr_features + [target_col]]

    return filtered_df





if __name__ == "__main__":
    # Export cleaned dataset
    clean_df = pd.read_csv('./dataset/clean_dataset.csv')
    target_col = 'ViolentCrimesPerPop'
    # Calculate correlation with target (excluding the target itself)
    correlation_matrix = clean_df.corr()[target_col].drop(target_col)

    # Convert to DataFrame and sort
    correlation_df = correlation_matrix.sort_values()

    # Plot
    plt.figure(figsize=(15, 6))  # WIDE plot to fit all 128 attribute names
    sns.set_style("whitegrid")
    plt.bar(correlation_df.index, correlation_df.values, color='skyblue', edgecolor='skyblue')

    plt.xticks(rotation=90, fontsize=8)  # Rotate attribute names for visibility
    plt.xlabel("Variable Name", fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel("Correlation Coefficient", fontsize=14, fontweight='bold', labelpad=5)
    plt.title(f'Correlation of Each Attribute with {target_col}')
    plt.axhline(0, color='gray', linestyle='--')  # horizontal line at 0
    plt.tight_layout()
    plt.savefig("graph/correlation_coefficient.png", dpi=300, bbox_inches='tight')  # save before plt.show()