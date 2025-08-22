from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ValidationCurveDisplay
import numpy as np
import train_model


def plot_validation_curve(model, X, Y, name, range, model_name):
    X, Y = shuffle(X, Y, random_state=42)
    ValidationCurveDisplay.from_estimator(estimator=model,X=X,y=Y,param_name=name, param_range=range, cv=10,scoring='accuracy')

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
            'name': 'SVC',
            'model': SVC(C=0.1, kernel='linear'),
            'params': {
                'C': np.arange(0.1, 2, 0.05),
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



