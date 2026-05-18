import pandas as pd
from pandas.api.types import is_numeric_dtype

def preprocessing(df):
    df = df.copy()

    # leakage
    df.drop(columns=['Claim_Status','Approved_Amount','Number_of_Claims_Per_Provider_Monthly'], inplace=True)
    
    # datetime
    df['Claim_Submission_Date'] = pd.to_datetime(df['Claim_Submission_Date'])
    df['Claim_Submission_Year'] = (df['Claim_Submission_Date'].dt.year)
    df['Claim_Submission_Day'] = (df['Claim_Submission_Date'].dt.dayofweek)
    df['Claim_Submission_Month'] = (df['Claim_Submission_Date'].dt.month)
    
    #missing values
    for col in df.columns[df.isnull().any()]:
        if is_numeric_dtype(df[col]):
            df[col + "_missing"] = (df[col].isnull().astype(int))
            df[col] = (df[col].fillna(df[col].median()))
        else: df[col] = (df[col].fillna('Unknown'))

    return df

def prepare_data(df):
    df = df.copy()

    train_df = df[df['Claim_Submission_Year'].isin([2021, 2022])].copy()
    val_df = df[df['Claim_Submission_Year'] == 2023].copy()
    test_df = df[df['Claim_Submission_Year'] == 2024].copy()

    X_train = train_df.drop('Is_Fraud', axis=1)
    Y_train = train_df['Is_Fraud']

    X_val = val_df.drop('Is_Fraud', axis=1)
    Y_val = val_df['Is_Fraud']

    X_test = test_df.drop('Is_Fraud', axis=1)
    Y_test = test_df['Is_Fraud']

    #frequency encoding
    freq_cols = [
        'Provider_ID',
        'Diagnosis_Code',
        'Procedure_Code'
    ]

    for col in freq_cols:
        freq_map = (X_train[col].value_counts(normalize=True))

        X_train[col + '_freq'] = (X_train[col].map(freq_map))
        X_val[col + '_freq'] = (X_val[col].map(freq_map))
        X_test[col + '_freq'] = (X_test[col].map(freq_map))
        
        X_val[col + '_freq'] = (X_val[col + '_freq'].fillna(0))
        X_test[col + '_freq'] = (X_test[col + '_freq'].fillna(0))

    cols_to_drop = [
        'Provider_ID',
        'Claim_ID',
        'Diagnosis_Code',
        'Procedure_Code',
        'Claim_Submission_Date'
    ]

    X_train.drop(columns=cols_to_drop, inplace=True)
    X_val.drop(columns=cols_to_drop, inplace=True)
    X_test.drop(columns=cols_to_drop, inplace=True)

    #one hot encoding
    cols_to_dummy = [
        'Patient_Gender',
        'Insurance_Type',
        'Provider_Specialty',
        'Patient_State',
        'Visit_Type'
    ]

    X_train = pd.get_dummies( X_train, columns=cols_to_dummy, dtype=int)
    X_val = pd.get_dummies(X_val, columns=cols_to_dummy, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cols_to_dummy, dtype=int)

    X_val = X_val.reindex(columns=X_train.columns,fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns,fill_value=0)

    return (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test
    )

