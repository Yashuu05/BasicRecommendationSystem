import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

def save_model(model, path):
    """
    function to save model in specific path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    
def load_model(path):
    """
    load the saved weights
    """
    return joblib.load(path)

def evaluate_regression_model(y_test, y_pred):
    """
    This function estimate the performance metrics of a given trained model
    Input: y_pred and y_test
    Output: performance metrics
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return mae, mse, rmse

def evaluate_classification_model(model, X_test, y_test):
    """
    this function calculate performance of the trained classification model
    Input: model, X_test and y_test
    Output: accuracy, f1 score, recall score and roc auc score
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Use probability predictions for ROC AUC when available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba, average='weighted')
    except:
        # Fallback to predictions if probabilities not available
        roc = roc_auc_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, recall, roc
    
def best_model(model_lst, peformance_lst, metric, type):
    """
    this function estimates best model on the basis of the given scoring metric
    """
    try:

        metric = metric.lower()
        if type=="regression":
            if metric in ['mae','mse','rmse']:
                best_score = peformance_lst[metric].min()
                model_name = peformance_lst.loc[peformance_lst[metric] == best_score, 'model_name'].iloc[0]
                best_model_obj = model_lst[model_name]
                return best_score, model_name, best_model_obj
            else:
                print(f"Given metric {metric} is not valid. Use mae, mse and rmse")
        elif type == "classification":
            if metric in ["accuracy","f1_score", "recall_score", "roc_auc_score"]:
                best_score = peformance_lst[metric].max()
                model_name = peformance_lst.loc[peformance_lst[metric] == best_score, "model_name"].iloc[0]
                best_model_obj = model_lst[model_name]
                return best_score, model_name, best_model_obj
            else:
                print(f"Given metric {metric} is not valid")
        else:
            print("Error: Type is invalid. Expected regression or classification.")

    except Exception as e:
        print(f"error: {e}")