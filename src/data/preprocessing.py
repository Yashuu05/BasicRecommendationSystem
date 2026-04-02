def clean_data(df):
    # handeling missing values:
    null_cols = []
    for cols in df.columns:
        if df[cols].isnull().sum() > 0:
            null_cols.append(cols)
        
    for cols in null_cols:
        if df[cols].dtypes == "object":
            df[cols] = df[cols].fillna(df[cols].mode())
        else:
            df[cols] = df[cols].fillna(df[cols].median())
    
    return null_cols, df 