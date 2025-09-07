import learning_curve
import train_model
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":

    X, Y = train_model.load_dataset()

    param_grid = {
        "n_neighbors": [27, 28, 29, 30, 31],
        "weights": ["uniform"],
        "algorithm": ["auto"],
        "leaf_size": [10,20,30,40,50],
        "metric" : ["minkowski"],
    }

    selected_model = train_model.model_evaluation(KNeighborsRegressor(), param_grid, X, Y)

    learning_curve.plot_learning_curve(selected_model.best_estimator_, X, Y, "KNR_test")

    train_model.save_model(selected_model.best_estimator_, "KNRmodel")

