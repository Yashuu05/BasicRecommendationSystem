import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipelines.model_pipelines import prepare_model_pipeline
from src.utils.data_utils import load_data, save_data, prepare_data_for_split, split_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from src.utils.model_utils import save_model, evaluate_model

def train_model(model_lst, X_train, y_train, X_test, y_test):
    # dictionary to store peformance of each model
    performance = {
    "model_name": [],
    "mae": [],
    "mse": [],
    "rmse": []
    }

    plt.figure(figsize=(8, 6))
    
    # train each model from the "model_lst"
    for name, model in model_lst.items():
        print(f"training {name}...")
        # fit the model
        model.fit(X_train, y_train)
        # predict target
        print("predicting target...")
        y_pred = model.predict(X_test)
        # evaluate performance
        mae, mse, rmse = evaluate_model(y_test=y_test, y_pred=y_pred)
        # store the metrics into performance list
        performance["model_name"].append(name)
        performance["mae"].append(mae)
        performance["mse"].append(mse)
        performance["rmse"].append(rmse)
        # plot scatter plot
        plt.scatter(y_test, y_pred, label=name, alpha=0.6)

    # Perfect prediction line
    plt.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--', linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (All Models)")
    plt.legend()
    # save the image
    image_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'results', 'performance.png')
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'results', 'model_performance.csv')
    plt.savefig(image_file_path)
    # save the performance od trained model
    performance = pd.DataFrame(performance, columns=['model_name','mae','mse','rmse'])
    performance.to_csv(file_path, index=False)


if __name__ == "__main__":
    # load dataset
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'clean_price_prediction.csv')
    df = load_data(path=file_path)
    # create "X" and "y" 
    X, y = prepare_data_for_split(data=df, target="price")
    # split dataset
    X_train, X_test, y_train, y_test = split_dataset(X=X, y=y, random_state=42, test_split_size=0.22)
    # ordinal cols
    ordinal_cols = ['severity','urgent','city_tier']
    # non_ordinal
    non_ordinal_cols = ['issue','device','brand','service_type','warranty_status']
    # num cols
    num_cols = ['device_age_years','technician_experien']
    # build pipeline
    model_lst = prepare_model_pipeline(
        non_ordinal_cols=non_ordinal_cols,
        ordinal_cols=ordinal_cols,
        num_cols=num_cols)
    # train model
    train_model(model_lst=model_lst, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)