import learning_curve
import train_model
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # Define the parameter grid for tuning
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  
        'max_depth': [2], # [None, 5, 10, 20]
        'min_samples_split': [5], #[8, 9, 10]
        'min_samples_leaf': [8], # [2, 3, 4]
        'max_features': [None, 'sqrt', 'log2'],
    }

    # Evaluate model
    selected_model = train_model.model_evaluation(DecisionTreeRegressor(random_state=42), param_grid, X, Y)

    # Plot the learning curve
    learning_curve.plot_learning_curve(selected_model.best_estimator_, X, Y, 'Decision_Tree_Test')

    # Save the selected model
    train_model.save_model(selected_model.best_estimator_, "DecisionTree_regression")





