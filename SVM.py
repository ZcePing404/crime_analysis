import learning_curve
import train_model
from sklearn.svm import SVC

if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # Define the parameter grid for tuning
    param_grid = {
        'C': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto']  
    }

    # Evaluate model
    selected_model = train_model.model_evaluation(SVC(), param_grid, X, Y)

    # Plot the learning curve
    learning_curve.plot_learning_curve(selected_model.best_estimator_, X, Y, 'SVM_test')

    # Save the selected model
    train_model.save_model(selected_model.best_estimator_, "SVM_classifier")





