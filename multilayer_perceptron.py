from train_model import load_dataset, model_evaluation, save_model
from learning_curve import plot_learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from keras.src.callbacks import EarlyStopping
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Input
from keras.src.optimizers import Adam, RMSprop, SGD
from scikeras.wrappers import KerasRegressor




# ===============================
# 1. Load Dataset
# ===============================
X, Y = load_dataset()

# ===============================
# 2. Train-validation-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ===============================
# 3. Normalize (important for NN!)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 4. Build Neural Network
# ===============================

# Define the parameter grid for tuning
param_dist = {
    'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__n_hidden': [1, 2, 3],         # number of hidden layers
    'model__n_neurons': [32, 64, 128],    # neurons per layer
    'model__activation': ['relu', 'tanh'],
    'model__dropout_rate': [0.0, 0.2, 0.5],
    "batch_size": [16, 32, 64],
}

def build_model(optimizer='adam', learning_rate=0.001, 
                n_hidden=2, n_neurons=64, activation='relu', 
                dropout_rate=0.2):
    model = Sequential([Input(shape=(X_train.shape[1],)), 
                        Dense(n_neurons, activation=activation),
                        Dropout(dropout_rate)])
    
    # Additional hidden layers
    for _ in range(n_hidden - 1):
        model.add(Dense(n_neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    
    # Output layer (regression case)
    model.add(Dense(1, activation='linear'))
    
    # Compile
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    return model

regressor  = KerasRegressor(model=build_model, epochs=50, batch_size=32, verbose=0)

best_model = model_evaluation(regressor, param_dist, X, Y, MLP=True)

plot_learning_curve(best_model.best_estimator_, X, Y, 'Multilayer Perceptron (MLP) test')

save_model(best_model.best_estimator_, "Multilayer_Perceptron")


# def hyperparameter_tuning_mlp(X, Y, regressor, param_dist, factor=3, resource='epochs', target='neg_mean_squared_error'):
#     halving_search = HalvingRandomSearchCV(estimator=regressor,
#                                            param_distributions=param_dist,
#                                            factor=factor,                   # how aggressively to cut configs (1/3 survive each round)
#                                            resource=resource,          # resource to allocate progressively
#                                            max_resources=100,          # max training epochs
#                                            min_resources=10,           # start small (10 epochs per model)
#                                            random_state=42,
#                                            cv=3,                       # 3-fold cross-validation
#                                            scoring=target,
#                                            verbose=1,
#                                            n_candidates='exhaust')      # try as many random candidates as possible)
#     halving_search.fit(X, Y)
#     return halving_search

# ===============================
# 5. Train Model
# ===============================
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )

