# Random Forest Hyperparameters
RF_PARAMS = {
"model__n_estimators": [100, 200, 300],
"model__max_depth": [10, 20, 30, None],
"model__min_samples_split": [2, 5],
"model__min_samples_leaf": [1, 2],
"model__max_features": ["sqrt", "log2"]
}

# Decision Tree Hyperparameters
DT_PARAMS = {
"model__max_depth": [None, 10, 20, 30],
"model__min_samples_split": [2, 5],
"model__min_samples_leaf": [1, 2],
"model__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
}

DT_CLASSIFICATION_PARAMS = {
"model__max_depth": [None, 10, 20, 30],
"model__min_samples_split": [2, 5],
"model__min_samples_leaf": [1, 2],
"model__criterion": ["gini", "entropy"],
"model__max_features": [None, "sqrt", "log2"]
}