
# Random Forest Hyperparameters
RF_PARAMS = {
"model__n_estimators": [50, 100, 200],
"model__max_depth": [None, 10, 20, 30],
"model__min_samples_split": [2, 5, 10],
"model__min_samples_leaf": [1, 2, 4]
}

# Decision Tree Hyperparameters
DT_PARAMS = {
"model__max_depth": [None, 5, 10, 20],
"model__min_samples_split": [2, 5, 10],
"model__min_samples_leaf": [1, 2, 4],
"model__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
}
