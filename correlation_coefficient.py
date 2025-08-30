import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


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



def filter_multicollinearity(df, target_col="ViolentCrimesPerPop"):
    # Separate features and target
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    # Fit Lasso with cross-validation to find best alpha
    lasso = LassoCV(cv=10, random_state=42).fit(X, Y)

    # Get coefficients
    coef = pd.Series(lasso.coef_, index=X.columns)

    # Display results
    print("\n\nBest alpha (λ) chosen by CV:", lasso.alpha_)
    print("\nNumber of selected features (non-zero coef):", sum(coef != 0))
    print("Number of removed features:", sum(coef == 0))

    print("\nRemoved Features:")
    print(coef[coef == 0].index.tolist())

    # Sort coefficients
    coef_sorted = coef.sort_values()

    # Split into chunks
    chunk_size = 51
    chunks = [coef_sorted[i:i+chunk_size] for i in range(0, len(coef_sorted), chunk_size)]

    for idx, chunk in enumerate(chunks):
        plt.figure(figsize=(14,6))
        chunk_colors = ["skyblue" if c != 0 else "lightgray" for c in chunk]
        chunk.plot(kind="bar", color=chunk_colors)

        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"Lasso Regression Feature Coefficients)")
        plt.ylabel("Coefficient Value")
        plt.xlabel("Features")
        plt.tight_layout()
        plt.savefig(f"graph/LassoRegression_{idx+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Keep only top 9 features by absolute coefficient value
    top_features = coef[coef != 0].abs().sort_values(ascending=False).head(9).index.tolist()
    filtered_df = df[top_features + [target_col]]

    return filtered_df


def plot_corr_coe(df, target_col = 'ViolentCrimesPerPop'):
    # Calculate correlation with target (excluding the target itself)
    correlation_matrix = df.corr()[target_col].drop(target_col)

    # Convert to DataFrame and sort
    correlation_df = correlation_matrix.sort_values()

    # Normalize correlation values to [0,1] for colormap
    norm = mcolors.Normalize(vmin=correlation_df.min(), vmax=correlation_df.max())
    cmap = cm.get_cmap("coolwarm")  # blue to red colormap
    colors = cmap(norm(correlation_df.values))

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))  
    sns.set_style("whitegrid")
    ax.bar(correlation_df.index, correlation_df.values, color=colors)

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



if __name__ == "__main__":
    df = pd.read_csv('./dataset/clean_dataset.csv')
    plot_corr_coe(df)