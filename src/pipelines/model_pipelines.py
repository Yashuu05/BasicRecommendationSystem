import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import src.pipelines.build_pipeline as build_pipeline
from xgboost import XGBRegressor, XGBClassifier
from src.config.model_params import DT_PARAMS, RF_PARAMS, DT_CLASSIFICATION_PARAMS
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def prepare_model_pipeline(non_ordinal_cols, ordinal_cols, num_cols, type, CV=5, scoring_param="roc_auc"):
    try:
        if type == "regression":
            # linear regression
            model = ElasticNet(random_state=42, alpha=1.0, l1_ratio=0.5,fit_intercept=True)
            model_pipeline = build_pipeline.create_pipeline(
                non_ordinal_cols=non_ordinal_cols,
                num_cols=num_cols,
                ordinal_cols=ordinal_cols,
                model=model
            )
        elif type == "classification":
            # SVM with probability for ROC AUC
            model = SVC(random_state=42, probability=True)
            model_pipeline = build_pipeline.create_pipeline(
                non_ordinal_cols=non_ordinal_cols,
                num_cols=num_cols,
                ordinal_cols=ordinal_cols,
                model=model
            )
        #decision tree 
        if type == "regression":
            dt_model = DecisionTreeRegressor()
            param = DT_PARAMS
        elif type == "classification":
            dt_model = DecisionTreeClassifier()
            param = DT_CLASSIFICATION_PARAMS
        # build pipeline
        dt_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=dt_model
        )
        dt_grid = GridSearchCV(
                estimator=dt_pipeline, param_grid=param, scoring=scoring_param, cv=CV, n_jobs=-1, verbose=2)
       
        # random forest
        if type == "regression":
            rf_model = RandomForestRegressor()
        elif type == "classification":
            rf_model = RandomForestClassifier()
        # build pipeline
        rf_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=rf_model
        )
        # create grid
        rf_grid = GridSearchCV(estimator=rf_pipeline, param_grid=RF_PARAMS, scoring=scoring_param, cv=CV, n_jobs=-1, verbose=2)

        # XGBoost
        if type == "regression":
            xgb_model = XGBRegressor()
        elif type == "classification":
            xgb_model = XGBClassifier()
        xgb_pipeline = build_pipeline.create_pipeline(
            non_ordinal_cols=non_ordinal_cols,
            num_cols=num_cols,
            ordinal_cols=ordinal_cols,
            model=xgb_model
        )

        model_lst = {
            "SVC": model_pipeline,
            "DecisionTree": dt_grid,
            "RandomForest": rf_grid,
            "XGBoost": xgb_pipeline
        }

        return model_lst
    
    except Exception as e:
        print(f"Error: {e}")
        return None
