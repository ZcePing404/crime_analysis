from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ValidationCurveDisplay
import numpy as np
import train_model


def plot_validation_curve(model, X, Y, name, range, model_name):
    X, Y = shuffle(X, Y, random_state=42)
    ValidationCurveDisplay.from_estimator(estimator=model,X=X,y=Y,param_name=name, param_range=range, cv=10,scoring='neg_mean_squared_error')

    plt.grid()
    plt.locator_params(axis='x', nbins=20)
    plt.title(f'Validation Curve of {model_name} with {name}')
    # Save the figure instead of showing it
    plt.savefig(f'./graph/vadCurve_{model_name}_{name}.png', dpi=600)
    print("Validation curve saved.")


if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # List of models and their parameter grids
    models = [
        {
            'name': 'SVR',
            'model': SVR(C=0.12, epsilon=0.06),
            'params': {
                'C': np.arange(0.01, 0.2, 0.01),
                'epsilon': np.arange(0.01, 0.2, 0.005)
            }
        },
        {
            'name': 'DecisionTree',
            'model':  DecisionTreeRegressor(max_depth=5, min_samples_leaf=7, min_samples_split=7, max_features='log2'),
            'params': {
                'max_depth': np.arange(1, 6, 1),            # tree depth from 1 to 20
                'min_samples_split': np.arange(5, 12, 1),    # node split requirement
                'min_samples_leaf': np.arange(1, 10, 1),     # leaf size
            }
        }
    ]

    for model in models:
        estimator = model['model']
        print(f"\nTraining {model['name']} model...")        
        print("\nModel:", estimator)

        params = model['params']

        for param_name, param_range in params.items():
            # Plot learning curve
            plot_validation_curve(estimator, X, Y, param_name, param_range, model['name']) 



