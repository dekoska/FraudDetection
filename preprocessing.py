import pandas as pd
from sklearn.model_selection import train_test_split

def preprocessing(df):
    df_copy = df.copy()

    nan_col = df_copy.columns[df_copy.isnull().any()]

    for col in nan_col:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].fillna('Unknown')
        else: df_copy[col] = df_copy[col].fillna(0)

    df_copy['Claim_Submission_Date'] = pd.to_datetime(df_copy['Claim_Submission_Date']) 

    df_copy['Claim_Submission_Day'] = df_copy['Claim_Submission_Date'].dt.dayofweek
    df_copy['Claim_Submission_Month'] = df_copy['Claim_Submission_Date'].dt.month
    df_copy['Claim_Submission_Year'] = df_copy['Claim_Submission_Date'].dt.year

    df_copy = df_copy.drop((['Claim_Submission_Date', 'Provider_ID', 'Claim_ID']), axis=1)

    #one hot encoding
    df_copy['Procedure_Code'] = df_copy['Procedure_Code'].astype(str)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    df_copy = pd.get_dummies(df_copy, columns=categorical_cols, dtype=int, drop_first=False) 

    return df_copy

def prepare_data(df):
    x = df.drop('Is_Fraud', axis=1)
    Y = df['Is_Fraud']

    x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=42, stratify=Y)
    return x_train, x_test, Y_train, Y_test


