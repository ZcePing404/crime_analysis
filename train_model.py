import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection._search_successive_halving import HalvingRandomSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import os
import pickle
#import smogn


def load_dataset(filename='final_dataset'):
    # Load preprocessed dataset
    try:
        df = pd.read_csv(f'./dataset/{filename}.csv')
    except FileNotFoundError:
        raise FileNotFoundError(f"The dataset file './dataset/{filename}.csv' was not found. Please check the file path.")
    # Clean column names
    df.columns = df.columns.str.strip()

    # # Apply SMOGN to training set only
    # resampled_df = smogn.smoter(data=df, y='ViolentCrimesPerPop')
    
    # Separate X and Y
    Y = df['ViolentCrimesPerPop']
    X = df.drop('ViolentCrimesPerPop', axis=1)

    return X, Y



def hyperparameter_tuning(X, Y, model, param_grid, target='neg_mean_squared_error', MLP=False):

    if MLP == False:
        grid = GridSearchCV(model, param_grid, scoring=target, cv=5, verbose=3)
        grid.fit(X, Y)
    else:
        grid = HalvingRandomSearchCV(estimator=model,
                                    param_distributions=param_grid,
                                    factor=3,                   
                                    resource="epochs",         
                                    max_resources=100,          
                                    min_resources=10,           
                                    random_state=42,
                                    cv=3,                       
                                    scoring=target,
                                    verbose=1,
                                    n_candidates='exhaust')
        grid.fit(X, Y)

    return grid



def model_evaluation(pipeline, param_grid, X, Y, MLP=False):
    # Split data for 80/20
    X_train_82, Y_train_82, X_test_82, Y_test_82 = split_data(X, Y, 0.2)
    model_82 = hyperparameter_tuning(X_train_82, Y_train_82, pipeline, param_grid, MLP=MLP)

    # Split data for 70/30
    X_train_73, Y_train_73, X_test_73, Y_test_73 = split_data(X, Y, 0.3)
    model_73 = hyperparameter_tuning(X_train_73, Y_train_73, pipeline, param_grid, MLP=MLP)

    #Prepare models and potential splits
    splits = [
        {"label": "80/20", "model": model_82}
    ]

    # only add 70/30 if its best_params differ
    if model_82.best_params_ != model_73.best_params_:
        splits.append({"label": "70/30", "model": model_73})

    # Test each split’s model and collect results
    results = {}
    for item in splits:
        lbl   = item["label"]
        mdl   = item["model"].best_estimator_
        results[lbl] = test_model(mdl, X, Y)

    # Select the best model by MSE
    best_label = min(results, key=lambda k: results[k]["MSE"])
    selected_model = model_82 if best_label == "80/20" else model_73

    # Display results for each split
    num_features = X.shape[1]
    for item in splits:
        lbl = item["label"]
        clf = item["model"].best_estimator_

        print(f"\n\nBest params from {lbl} training data with {num_features} features")
        print("Classifier : ", clf)
        print("-" * 81)
        print("10-Fold Cross Validation")
        print("-" * 81)
        display_result(results[lbl])

    print(f"\nSelected split: {best_label} (MSE = {results[best_label]['MSE']:.4f})")
    return selected_model





def split_data(X, Y, size):
    print("\n\nSpliting Data...")
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, Y_train, X_test, Y_test



def test_model(model, X, Y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Store results
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    Y_true_all = []
    Y_pred_all = []

    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # append results
        Y_true_all.extend(Y_test)
        Y_pred_all.extend(Y_pred)
        # Calculate metrics
        mse_scores.append(mean_squared_error(Y_test, Y_pred))
        rmse_scores.append(root_mean_squared_error(Y_test, Y_pred))
        mae_scores.append(mean_absolute_error(Y_test, Y_pred))
        r2_scores.append(r2_score(Y_test, Y_pred))

    return {
        'MSE': np.mean(mse_scores),
        'RMSE': np.mean(mse_scores),
        'MAE': np.mean(mae_scores),
        'R2': np.mean(r2_scores)
    }

    

def display_result(result):
    print(f"MSE: {result['MSE']:.4f}")
    print(f"RMSE: {result['RMSE']:.4f}")
    print(f"MAE: {result['MAE']:.4f}")
    print(f"R2: {result['R2']:.4f}")


def save_model(model, file_name):
    # Save model and vectorizer
    os.makedirs('./model', exist_ok=True)
    with open(f'./model/{file_name}.pkl', 'wb') as f:
        pickle.dump((model), f)

    print("Model saved successfully!")