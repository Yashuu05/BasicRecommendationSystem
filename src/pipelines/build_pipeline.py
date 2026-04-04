from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

def create_pipeline(non_ordinal_cols, num_cols, ordinal_cols, model):
    
    ordinal_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ordinalencode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])  

    non_ordinal_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy="most_frequent")),
        ('onehotencode', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('non_ordinal', non_ordinal_preprocessor, non_ordinal_cols),
        ('num', num_preprocessor, num_cols)
    ], remainder="passthrough")

    model_pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', model)
    ])

    return model_pipeline