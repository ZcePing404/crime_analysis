import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.utils import resample
from matplotlib_venn import venn2


def get_highly_correlated_features(df, threshold, target_col='ViolentCrimesPerPop',):
    # Calculate correlation
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_col].abs().sort_values()

    # Filter columns with high correlation (including the target itself)
    high_corr_features = target_corr[target_corr >= threshold].drop(target_col).index.tolist()

    print("\n\nCorrelation Coefficient > 0.2")
    print("Selected features:", high_corr_features)

    # Return a new DataFrame with only the selected features + target
    filtered_df = df[high_corr_features + [target_col]]

    return filtered_df



def plot_corr_coe(df, target_col = 'ViolentCrimesPerPop'):
    # Calculate correlation with target (excluding the target itself)
    correlation_matrix = df.corr()[target_col].drop(target_col)

    # Convert to DataFrame and sort
    correlation_df = correlation_matrix.sort_values()

    # Normalize correlation values to [0,1] for colormap
    norm = mcolors.Normalize(vmin=correlation_df.min(), vmax=correlation_df.max())
    cmap = plt.get_cmap("coolwarm")  # blue to red colormap
    colors = cmap(norm(correlation_df.values))

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))  
    sns.set_style("whitegrid")
    ax.bar(correlation_df.index, correlation_df.values, color=colors)

    ax.set_xticks(range(len(correlation_df.index)))
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








def mutual_information_test(df, target_col="ViolentCrimesPerPop", threshold=0.05):
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    # Compute MI scores
    mi_scores = mutual_info_regression(X, Y, random_state=42)

    # Put into dataframe
    mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values(by="MI_Score", ascending=False)

    # Filter features above threshold
    selected_df = mi_df[mi_df["MI_Score"] > threshold]

    print(f"Selected {len(selected_df["Feature"])} features out of {X.shape[1]}")

    plt.figure(figsize=(12,6))
    plt.bar(selected_df["Feature"], selected_df["MI_Score"], color="skyblue")
    plt.ylabel("Mutual Information Score")
    plt.xlabel("Features")
    plt.title("Mutual Information between Features and Target")
    plt.xticks(rotation=90)  # rotate labels for readability
    plt.tight_layout()
    plt.savefig("graph/mutual_info.png", dpi=300)
    plt.close()

    
    filtered_df = df[selected_df["Feature"].tolist() + [target_col]]

    return filtered_df








def rfe_test_n_features(df, target_col="ViolentCrimesPerPop"):
    # Split features and target
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    feature_names = X.columns

    # RFE with Elastic Net
    print(f"\nRFE with Elastic Net Testing N features")
    model = SVR(kernel="linear").fit(X, Y)
    scores = []

    # try RFE for different numbers of features
    for n_features in range(5, min(30, len(feature_names))):  # try from at least 5 features
        print(f"Testing n_features : {n_features}")
        rfe = RFE(model, n_features_to_select=n_features)
        X_rfe = rfe.fit_transform(X, Y)
        
        # cross_val_score returns negative MSE, so take mean RMSE
        rmse = np.sqrt(-cross_val_score(rfe, X_rfe, Y, cv=10, scoring="neg_mean_squared_error"))
        mean_rmse = round(rmse.mean(), 4)
        scores.append((n_features, mean_rmse))

    
    results = np.array(scores)
    feature_counts, mean_rmses = results[:,0], results[:,1]

    # Plot RMSE vs number of features
    plt.figure(figsize=(10,6))
    plt.errorbar(feature_counts, mean_rmses, fmt='-o', capsize=3)
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE")
    plt.title("RFE Feature Selection")
    plt.savefig("graph/RFE_performance.png", dpi=300)
    plt.close()



def rfe_test(df, n_features, target_col="ViolentCrimesPerPop"):
    # Split features and target
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    # Use Elastic Net as base estimator
    model = SVR(kernel="linear").fit(X, Y)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, Y)

    # Get selected feature names
    selected_features = X.columns[rfe.support_]
    coefs = pd.Series(rfe.estimator_.coef_.ravel(), index=selected_features)

    # Pick top N by absolute coefficient
    top_features = coefs.abs().sort_values(ascending=False).head(n_features)

    plt.figure(figsize=(10,6))
    top_features.plot(kind="barh", color="skyblue")
    plt.xlabel("Variable Importance")
    plt.title(f"Features Selected by RFE")
    plt.gca().invert_yaxis()
    plt.savefig("graph/RFE_top_features.png", dpi=300)
    plt.close()

    return top_features.index.to_list()





def elastic_net_test(df, alpha, target_col="ViolentCrimesPerPop"):
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    feature_names = X.columns

    # Fit Elastic Net CV
    enet = ElasticNet(alpha=alpha, max_iter=10000, random_state=42).fit(X, Y)

    coefs = pd.Series(enet.coef_, index=feature_names)
    importance = coefs[coefs != 0].abs().sort_values(ascending=False).index.to_list()


    plt.figure(figsize=(14,6))
    coefs.abs().sort_values(ascending=False).plot(kind="bar", color="skyblue")

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Features by Elastic Net")
    plt.ylabel("Coefficients")
    plt.xlabel("Features")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig("graph/Elastic_net.png", dpi=300)
    plt.close()

    return importance




def elastic_net_test_alpha(df, target_col="ViolentCrimesPerPop"):
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    # Fit Elastic Net CV
    enet = ElasticNetCV(l1_ratio=0.5, cv=10, max_iter=10000, random_state=42).fit(X, Y)

    # ----- 1-SE RULE -----
    # Average MSE across bootstraps
    mse_mean = enet.mse_path_.mean(axis=1)
    mse_std = enet.mse_path_.std(axis=1)

    # Best alpha
    best_idx = np.argmin(mse_mean)
    best_alpha = enet.alphas_[best_idx]
    best_error = mse_mean[best_idx]

    # 1-SE alpha: largest alpha with error <= best_error + std
    one_se_error = best_error + mse_std[best_idx]

    # Find closest alpha satisfying 1-SE rule
    candidate_alphas = enet.alphas_[mse_mean <= one_se_error]
    one_se_alpha = candidate_alphas.max() if len(candidate_alphas) > 0 else best_alpha

    print(f"Best alpha (λ): {best_alpha}")
    print(f"1-SE alpha (λ): {one_se_alpha}")

    # ----- PLOT -----
    plt.figure(figsize=(10,6))
    plt.errorbar(np.log(enet.alphas_), mse_mean, yerr=mse_std, fmt='o', ecolor='lightgray', elinewidth=2, capsize=4, color='red')
    plt.axvline(np.log(best_alpha), linestyle="--", color="blue", label=f"Best α = {best_alpha:.4f}")
    plt.axvline(np.log(one_se_alpha), linestyle="--", color="green", label=f"1-SE α = {one_se_alpha:.4f}")

    plt.xlabel("Log(Alpha)")
    plt.ylabel("Mean Squared Error")
    plt.title("Elastic Net CV with 1-SE Rule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graph/ElasticNet_alpha.png", dpi=300)
    plt.close()

    return one_se_alpha




def plot_venn(df, elastic_features, rfe_features, target_col="ViolentCrimesPerPop"):
    # ---------------------------
    # 4. Venn Diagram (Intersection)
    # ---------------------------
    enet_set = set(elastic_features)
    rfe_set = set(rfe_features)
    intersection = list(enet_set & rfe_set)

    plt.figure(figsize=(6,6))
    venn2([enet_set, rfe_set], set_labels=("Elastic Net", "RFE"))
    plt.title("Feature Overlap between Elastic Net and RFE")
    plt.savefig("graph/feature_venn.png", dpi=300)
    plt.close()

    print(f"Intersection features ({len(intersection)}):", intersection)    
    return df[intersection + [target_col]]




if __name__ == "__main__":
    df = pd.read_csv('./dataset/clean_dataset.csv')
    plot_corr_coe(df)

    # Select only highly related features
    before = df.shape[1]
    # plot_corr_coe(df)
    # df = get_highly_correlated_features(df, threshold=0.2)
    df = mutual_information_test(df)
    after = df.shape[1]
    print(f"\nMutual Information Testing")
    print(f"After remove MI < 0.05 : {after} attributes (Removed {before - after})\n\n")

    # Elastic Net Test
    print(f"\nElastic Net Testing")
    before = df.shape[1]
    alpha = elastic_net_test_alpha(df)
    ent_selected_features = elastic_net_test(df, alpha)

    # RFE Test
    print(f"\nRFE Testing")
    rfe_test_n_features(df)
    rfe_selected_features = rfe_test(df, n_features=17)

    print(f"\nElastic Net selected features {len(ent_selected_features)} : {ent_selected_features}")
    print(f"RFE selected features {len(rfe_selected_features)} : {rfe_selected_features}")

    df[ent_selected_features + ["ViolentCrimesPerPop"]].to_csv('./dataset/elastic_net_dataset.csv', index=False)
    df[rfe_selected_features + ["ViolentCrimesPerPop"]].to_csv('./dataset/rfe_dataset.csv', index=False)

    df = plot_venn(df, ent_selected_features, rfe_selected_features)
    after = df.shape[1]
    print(f"\nAfter Combining two method : {after} attributes")


    print(f"\nFinal number of features        : {after}")
    print(df.columns.tolist())
    df.to_csv('./dataset/final_dataset.csv', index=False)