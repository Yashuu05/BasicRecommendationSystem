import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    function to read dataset
    """

    try:
        print("reading dataset...")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_data(data, save_file_path):
    """
    function to save the dataset
    """

    try:
        print("saving data...")
        data.to_csv(save_file_path, index=False)
        print(f"{data} saved successfully")
    except Exception as e:
        print(f"Error: {e}")

def prepare_data_for_split(data, target):
    """
    This function returns labels and target feature from given dataset
    Input: dataset (e.g data.csv)
    Output: X (input labels), y (target feature)
    """
    y = data[f"{target}"]
    X = data.drop(f"{target}", axis=1)
    return X, y

def split_dataset(X,y,random_state, test_split_size):
    """
    This function splits the dataset into train and test dataset
    Input: processed data (data.csv)
    output: splitted datasets (X_train, X_test, y_test, y_train)
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=random_state, test_size=test_split_size, shuffle=True)
    return X_train, X_test, y_train, y_test