import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def get_highly_correlated_features(df, threshold, target_col='ViolentCrimesPerPop',):
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
    # Export cleaned dataset
    clean_df = pd.read_csv('./dataset/clean_dataset.csv')
    target_col = 'ViolentCrimesPerPop'

    # Calculate correlation with target (excluding the target itself)
    correlation_matrix = clean_df.corr()[target_col].drop(target_col)

    # Convert to DataFrame and sort
    correlation_df = correlation_matrix.sort_values()

    # Normalize correlation values to [0,1] for colormap
    norm = mcolors.Normalize(vmin=correlation_df.min(), vmax=correlation_df.max())
    cmap = cm.get_cmap("coolwarm")  # blue to red colormap
    colors = cmap(norm(correlation_df.values))

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))  
    sns.set_style("whitegrid")
    bars = ax.bar(correlation_df.index, correlation_df.values, color=colors)

    ax.set_xticklabels(correlation_df.index, rotation=90, fontsize=8)  
    ax.set_xlabel("Variable Name", fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel("Correlation Coefficient", fontsize=14, fontweight='bold', labelpad=5)
    ax.set_title(f'Correlation of Each Attribute with {target_col}')
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--')  

    # Add colorbar to the figure explicitly
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", label="Correlation Strength")

    plt.tight_layout()
    plt.savefig("graph/correlation_coefficient.png", dpi=300, bbox_inches='tight')