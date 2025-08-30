from sklearn.svm import SVR

import learning_curve
import train_model
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":

    X, Y = train_model.load_dataset()

    param_grind = {
        "n_neighbors": [],
        "weights" : ['uniform', 'distance'],
    }

    selected_model = train_model.model_evaluation(KNeighborsRegressor(), param_grind, X, Y)

    learning_curve.plot_learning_curve(selected_model.best_estimator_, X, Y, "KNR_test")

    train_model.save_model(selected_model.best_estimator_, "KNRmodel")

