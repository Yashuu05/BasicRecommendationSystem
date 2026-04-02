import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.preprocessing import clean_data
from src.utils.data_utils import load_data
from src.utils.data_utils import save_data

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw','recommendation_dataset.csv')
    save_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed','clean_recommendation_dataset.csv')
    data = load_data(path=file_path)
    print("=============== Before cleaning ==============")
    print("Null values = \n", data.isnull().sum())
    print("total null values = ", data.isnull().sum().sum())
    print("total duplicate values = ", data.duplicated().sum())
    print(f"total rows : {data.shape[0]} |  total columns : {data.shape[1]}")

    # clean data
    null_cols, df = clean_data(df=data)
    print("=============== After cleaning ==============")
    print("null columns =", null_cols)
    print("Null values = \n", df.isnull().sum())
    print("total null values = ", df.isnull().sum().sum())
    print("total duplicate values = ", df.duplicated().sum())
    print(f"total rows : {df.shape[0]} |  total columns : {df.shape[1]}")
    # save the cleaned dataset
    save_data(data=df, save_file_path=save_file_path)