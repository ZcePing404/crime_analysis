import train_model
from sklearn.svm import SVR

if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # Define the parameter grid for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],  
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto']  
    }


    # Apply split data in 8:2
    model_split82, X_test, Y_test = train_model.split_data(SVR(), param_grid, X, Y, 0.2)
    result_split82 = train_model.test_model(model_split82, X_test, Y_test)

    # Apply spit data in 7:3
    model_split73, X_test, Y_test = train_model.split_data(SVR(), param_grid, X, Y, 0.3)
    result_split73 = train_model.test_model(model_split73, X_test, Y_test)

    # Apply stratified 10-fold cross validation
    model_cv, result_cv = train_model.regression_cross_validation(SVR(),  param_grid, X, Y)

    print("\n\nSplit Data 80/20")
    print("---------------------------------------------------------------------------------")
    print("Algorithm :", model_split82)
    print("---------------------------------------------------------------------------------")
    train_model.display_result(result_split82)

    print("\n\nSplit Data 70/30")
    print("---------------------------------------------------------------------------------")
    print("Algorithm :", model_split73)
    print("---------------------------------------------------------------------------------")
    train_model.display_result(result_split73)

    print("\n\nRegreesion 10-Fold Cross Vaidation")
    print("---------------------------------------------------------------------------------")
    print("Algorithm :", model_cv)
    print("---------------------------------------------------------------------------------")
    train_model.display_result(result_cv)
    print("\n\n")

    train_model.save_model(model_split82)





