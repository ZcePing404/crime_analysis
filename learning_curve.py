from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import LearningCurveDisplay
import numpy as np
import train_model


def plot_learning_curve(model, X, Y, model_name):
    X, Y = shuffle(X, Y, random_state=0)
    LearningCurveDisplay.from_estimator(estimator=model,X=X,y=Y,train_sizes=np.linspace(0.1, 1.0, 10), cv=10, scoring='neg_mean_squared_error')

    plt.grid()
    plt.title(f'Learning Curve of {model_name}')
    # Save the figure instead of showing it
    plt.savefig(f'./graph/learning_curve_{model_name}.png', dpi=300)
    print("Learning curve saved.")


if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # List of models and their parameter grids
    model_and_params = [
        {
            'name': 'SVM',
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 0.12, 0.14],
                'epsilon': [0.02, 0.04, 0.06],  
                'kernel': ['linear', 'rbf'], 
                'gamma': ['scale', 'auto']  
            }
        },
        {
            'name': 'DecisionTree',
            'model': DecisionTreeRegressor(),
            'param_grid': {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  
                'max_depth': [2, 3, 4],
                'min_samples_split': [7, 8, 9],
                'min_samples_leaf': [2, 3, 4],
                'max_features': [None, 'sqrt', 'log2'],
            }
        }
        # other models
    ]

    for config in model_and_params:
        print(f"\nTraining {config['name']} model...")
        model = train_model.hyperparameter_tuning(X, Y, config['model'], config['param_grid'])
    
        print("Model:", model.best_estimator_)
        # Plot learning curve
        plot_learning_curve(model.best_estimator_, X, Y, config['name']) 



