import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

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

def evaluate_model(y_test, y_pred):
    """
    This function estimate the performance metrics of a given trained model
    Input: trained model
    Output: performance metrics
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return mae, mse, rmse
    
def best_model(model_lst, peformance_lst, metric):
    """
    this function estimates best model on the basis of the given scoring metric
    """
    try:
        metric = metric.lower()
        if metric in ['mae','mse','rmse']:
            best_score = peformance_lst[metric].min()
            model_name = peformance_lst.loc[peformance_lst[metric] == best_score, 'model_name'].iloc[0]
            best_model_obj = model_lst[model_name]
            return best_score, model_name, best_model_obj
        else:
            print(f"Given metric {metric} is not valid. Use mae, mse and rmse")

    except Exception as e:
        print(f"error: {e}")