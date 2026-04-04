import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import pandas as pd

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
        return None

def predict_price(input_data, model):
    """
    This function is responsible for predicting price 
    Input: user input
    output: predicted price
    """
    input_data = pd.DataFrame([input_data])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

def predict_recommendation(input_data, model, top_k=3):
    """
    Predict and rank service providers based on input features.
    Args:
        input_data (pd.DataFrame): Data containing provider + user features
        model_path (str): Path to trained model
        top_k (int): Number of top providers to return
    Returns:
        pd.DataFrame: Top ranked providers
    """

    # Copy Data 
    df = input_data.copy()

    # Save provider name separately
    provider_names = df["provider_name"]

    # Drop non-ML columns
    if "provider_name" in df.columns:
        df = df.drop(columns=["provider_name"], axis=1)

    expected_columns = [
        "issue", "device", "severity", "urgent",
        "rating", "num_reviews", "success_rate", "experience_years",
        "distance_km", "response_time_min", "base_price",
        "service_type", "availability"
    ]
    df = df[expected_columns]

    # Predict Probabilities
    scores = model.predict_proba(df)[:, 1]

    # Attach Scores
    result_df = input_data.copy()
    result_df["score"] = scores

    # Sort Providers
    ranked = result_df.sort_values(by="score", ascending=False)

    # Return Top K
    return ranked.head(top_k).reset_index(drop=True)