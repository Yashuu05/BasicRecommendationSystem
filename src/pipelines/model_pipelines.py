import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import src.pipelines.build_pipeline as build_pipeline
from xgboost import XGBRegressor
from src.config.model_params import DT_PARAMS, RF_PARAMS
from sklearn.model_selection import GridSearchCV

def prepare_model_pipeline(non_ordinal_cols, ordinal_cols, num_cols, CV=5, scoring_param='neg_mean_squared_error'):
    try:
        # linear regression
        en_model = ElasticNet(random_state=42, alpha=1.0, l1_ratio=0.5,fit_intercept=True)
        en_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=en_model
        )

        #decision tree 
        dt_model = DecisionTreeRegressor()
        # build pipeline
        dt_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=dt_model
        )
        # create grid 
        dt_grid = GridSearchCV(
            estimator=dt_pipeline, param_grid=DT_PARAMS, scoring=scoring_param, cv=CV, n_jobs=-1, verbose=2)

        # random forest
        rf_model = RandomForestRegressor()
        # build pipeline
        rf_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=rf_model
        )
        # create grid
        rf_grid = GridSearchCV(estimator=rf_pipeline, param_grid=RF_PARAMS, scoring=scoring_param, cv=CV, n_jobs=-1, verbose=2)

        xgb_model = XGBRegressor()
        xgb_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=xgb_model
        )

        model_lst = {
            "ElasticNet": en_pipeline,
            "DecisionTreeRegressor": dt_grid,
            "RandomForest": rf_grid,
            "XGBoost": xgb_pipeline
        }

        return model_lst
    
    except Exception as e:
        print(f"Error: {e}")
        return None
