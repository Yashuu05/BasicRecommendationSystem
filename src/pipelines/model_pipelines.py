import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.pipelines import build_pipeline
from xgboost import XGBRegressor

def prepare_model_pipeline(non_ordinal_cols, ordinal_cols, num_cols):
    try:
        # linear regression
        en_model = ElasticNet()
        en_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=en_model
        )

        #decision tree 
        dt_model = DecisionTreeRegressor()
        dt_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=dt_model
        )

        # random forest
        rf_model = RandomForestRegressor()
        rf_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=rf_model
        )

        xgb_model = XGBRegressor()
        xgb_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=xgb_model
        )

        model_lst = {
            "ElasticNet": en_pipeline,
            "DecisionTreeRegressor" : dt_pipeline,
            "RandomForest": rf_pipeline,
            "XGBoost": xgb_pipeline
        }

        return model_lst
    
    except Exception as e:
        print(f"Error: {e}")
        return None
