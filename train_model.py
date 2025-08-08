import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import os
import pickle
import smogn


def load_dataset():
    # Load preprocessed dataset
    try:
        df = pd.read_csv('./dataset/final_dataset.csv')
    except FileNotFoundError:
        raise FileNotFoundError("The dataset file './dataset/final_dataset.csv' was not found. Please check the file path.")
    # Clean column names
    df.columns = df.columns.str.strip()

    # # Apply SMOGN to training set only
    # resampled_df = smogn.smoter(data=df, y='ViolentCrimesPerPop')
    
    # Separate X and Y
    Y = df['ViolentCrimesPerPop']
    X = df.drop('ViolentCrimesPerPop', axis=1)

    return X, Y



def hyperparameter_tuning(X, Y, model, param_grid, target='neg_mean_squared_error'):
    grid = GridSearchCV(model, param_grid, scoring=target, cv=5, n_jobs=-1, verbose=3)
    grid.fit(X, Y)
    return grid


# def regression_cross_validation(algorithm, param_grid, X, Y):
#     kf = KFold(n_splits=10, shuffle=True, random_state=42)

#     # Store results
#     mse_scores = []
#     rmse_scores = []
#     mae_scores = []
#     r2_scores = []

#     # GridSearch for best model on the whole training set
#     grid = GridSearchCV(algorithm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid.fit(X, Y)
#     best_model = grid.best_estimator_

#     for train_idx, test_idx in kf.split(X):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

#         model_cv = clone(best_model)
#         model_cv.fit(X_train, Y_train)
#         Y_pred = model_cv.predict(X_test)

#         # Calculate metrics
#         mse_scores.append(mean_squared_error(Y_test, Y_pred))
#         rmse_scores.append(root_mean_squared_error(Y_test, Y_pred))
#         mae_scores.append(mean_absolute_error(Y_test, Y_pred))
#         r2_scores.append(r2_score(Y_test, Y_pred))

#     # Report mean results
#     return model_cv, {
#         'MSE': np.mean(mse_scores),
#         'RMSE': np.mean(mse_scores),
#         'MAE': np.mean(mae_scores),
#         'R2': np.mean(r2_scores)
#     }




def split_data(X, Y, size):
    print("\n\nSpliting Data...")
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, Y_train, X_test, Y_test



def test_model(model, X_test, Y_test):
    # Predict on test set
    Y_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = root_mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    

def display_result(result):
    print(f"MSE: {result['MSE']:.4f}")
    print(f"RMSE: {result['RMSE']:.4f}")
    print(f"MAE: {result['MAE']:.4f}")
    print(f"R2: {result['R2']:.4f}")


def save_model(model):
    # Save model
    os.makedirs('./model', exist_ok=True)
    with open('./model/model.pkl', 'wb') as f:
        pickle.dump((model), f)

    print("Model saved successfully!")