import learning_curve
import train_model
from sklearn.svm import SVR

if __name__ == "__main__":
    # Load x and y from dataset
    X, Y = train_model.load_dataset()

    # Define the parameter grid for tuning
    param_grid = {
        'C': [0.1, 0.12, 0.14],
        'epsilon': [0.02, 0.04, 0.06],  
    }

    # Evaluate model
    selected_model = train_model.model_evaluation(SVR(), param_grid, X, Y)

    # Plot the learning curve
    learning_curve.plot_learning_curve(selected_model.best_estimator_, X, Y, 'SVM_test')

    # Save the selected model
    train_model.save_model(selected_model.best_estimator_, "SVM_classifier")





