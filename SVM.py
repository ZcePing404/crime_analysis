import numpy as np
from sklearn.model_selection import cross_val_score
import train_model
from sklearn.svm import SVR

if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # Define the parameter grid for tuning
    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'epsilon': [0.005, 0.01, 0.05, 0.1, 0.2],  
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto']  
    }


    # Apply split data in 80/20
    X_train, Y_train, X_test, Y_test = train_model.split_data(X, Y, 0.2)
    model_82 = train_model.hyperparameter_tuning(X_train, Y_train, SVR(), param_grid)
    result_split82 = train_model.test_model(model_82.best_estimator_, X_test, Y_test)

    # Apply spit data in 80/20
    X_train, Y_train, X_test, Y_test = train_model.split_data(X, Y, 0.3)
    model_73 = train_model.hyperparameter_tuning(X_train, Y_train, SVR(), param_grid)
    result_split73 = train_model.test_model(model_73.best_estimator_, X_test, Y_test)

    # Apply stratified 10-fold cross validation
    scores = cross_val_score(model_82.best_estimator_, X, Y, cv=10, scoring='neg_mean_squared_error')

    print("\n\nSplit Data 80/20")
    print("---------------------------------------------------------------------------------")
    print("Algorithm :", model_82.best_estimator_)
    print("---------------------------------------------------------------------------------")
    train_model.display_result(result_split82)

    print("\n\nSplit Data 70/30")
    print("---------------------------------------------------------------------------------")
    print("Algorithm :", model_73.best_estimator_)
    print("---------------------------------------------------------------------------------")
    train_model.display_result(result_split73)

    print("\n\nRegreesion 10-Fold Cross Vaidation")
    print("---------------------------------------------------------------------------------")
    # Convert negative mean squared error to positive for easier interpretation
    rmse_scores = np.sqrt(np.abs(scores))
    print(f"Average RMSE       : {np.mean(rmse_scores):.4f}")
    print(f"Standard deviation : {np.std(rmse_scores):.4f}")
    print("\n\n")

    train_model.save_model(model_82)





