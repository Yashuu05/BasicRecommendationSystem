import joblib
import os

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
