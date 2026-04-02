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
    
