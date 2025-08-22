import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import pickle


def load_dataset():
    # Load preprocessed dataset
    try:
        df = pd.read_csv('./dataset/final_dataset.csv')
    except FileNotFoundError:
        raise FileNotFoundError("The dataset file './dataset/final_dataset.csv' was not found. Please check the file path.")
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Separate X and Y
    Y = df['cid']
    X = df.drop('cid', axis=1)

    return X, Y



def hyperparameter_tuning(X, Y, model, param_grid, target='accuracy'):
    grid = GridSearchCV(model, param_grid, scoring=target, cv=5, verbose=3)
    grid.fit(X, Y)
    return grid


def test_model(model, X, Y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Store true and predicted labels across all folds
    accuracies = []
    Y_true_all = []
    Y_pred_all = []

    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # append results
        Y_true_all.extend(Y_test)
        Y_pred_all.extend(Y_pred)
        accuracies.append(accuracy_score(Y_test, Y_pred))


    # Margin of error (95% confidence)
    accuracy = np.mean(accuracies)
    std = np.std(accuracies)
    print(std)
    margin_error = 1.96 * (std / np.sqrt(len(accuracies)))

    # Generate classification report
    report = classification_report(Y_true_all, Y_pred_all, target_names=['0','1'])

    # Print Confusion Matrix
    cm = confusion_matrix(Y_true_all, Y_pred_all)

    return {
        'accuracy': accuracy,
        'margin_error': margin_error,
        'report': report,
        'cm': cm
    }


def model_evaluation(pipeline, param_grid, X, Y):
    # Split data for 80/20
    X_train_82, Y_train_82, X_test_82, Y_test_82 = split_data(X, Y, 0.2)
    model_82 = hyperparameter_tuning(X_train_82, Y_train_82, pipeline, param_grid)

    # Split data for 70/30
    X_train_73, Y_train_73, X_test_73, Y_test_73 = split_data(X, Y, 0.3)
    model_73 = hyperparameter_tuning(X_train_73, Y_train_73, pipeline, param_grid)

    #Prepare models and potential splits
    splits = [
        {"label": "80/20", "model": model_82}
    ]

    # only add 70/30 if its best_params differ
    if model_82.best_params_ != model_73.best_params_:
        splits.append({"label": "70/30", "model": model_73})

    # Test each split’s model and collect results
    results = {}
    for item in splits:
        lbl   = item["label"]
        mdl   = item["model"].best_estimator_
        results[lbl] = test_model(mdl, X, Y)

    # Select the best model by accuracy
    best_label = max(results, key=lambda k: results[k]["accuracy"])
    selected_model = model_82 if best_label == "80/20" else model_73

    # Display results for each split
    for item in splits:
        lbl = item["label"]
        clf = item["model"].best_estimator_

        print(f"\n\n10-Fold Cross Validation with {lbl} training data")
        print("-" * 81)
        print("Classifier : ", clf)
        print("-" * 81)
        display_result(results[lbl])

    print(f"\nSelected split: {best_label} (accuracy = {results[best_label]['accuracy']:.4f})")
    return selected_model




def split_data(X, Y, size):
    from collections import Counter
    print("\n\nSpliting Data...")
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=42)
    print('Original dataset shape %s' % Counter(Y))
    # Apply BorderlineSMOTE to training set only
    sm = BorderlineSMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    print('Resampled dataset shape %s' % Counter(Y_res))
    return X_res, Y_res, X_test, Y_test

    

def display_result(result):
    print(f"Avg Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"Margin error : {result['margin_error'] * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:\n")
    print(result['report'])

    # Print Confusion Matrix
    print("Confusion Matrix:\n", result['cm'])

    

def save_model(model, file_name):
    # Save model and vectorizer
    os.makedirs('./model', exist_ok=True)
    with open(f'./model/{file_name}.pkl', 'wb') as f:
        pickle.dump((model), f)

    print("Model saved successfully!")