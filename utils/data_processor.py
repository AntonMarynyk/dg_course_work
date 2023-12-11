import pandas as pd
import requests
from io import StringIO
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DataProcessor:
    def __init__(self, df=None) -> None:
        if df is not None:
            self.df = df

    def fetch_file(self, link):        
        r = requests.get(link)
        data = r.content.decode('utf-8')
        self.df = pd.read_csv(StringIO(data))
        return self.df
    
    def prepare_data(self):
        df_copy = self.df.copy()
        df_copy.drop(columns='customerID', inplace=True)
        df_copy.drop(columns='ChurnReason', inplace=True)
        df_copy.TotalCharges = pd.to_numeric(df_copy.TotalCharges, errors='coerce')
        df_copy.drop(columns='ZIPcode', inplace=True)

        if df_copy.duplicated().sum() > 0:
            df_copy.drop_duplicates(inplace=True)
            print("Duplicates removed")
        else:
            print("No duplicates found")
        
        for column in df_copy.columns:
            if df_copy[column].dtype == 'object':
                df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
            else:
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
        
        df_copy['MultipleLines'] = df_copy['MultipleLines'].replace('No phone service', 'No')
        df_copy[[
            'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection','TechSupport', 
            'StreamingTV', 'StreamingMovies'
        ]] = df_copy[[
            'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
            ]].replace('No internet service', 'No')

        df_copy['PaymentMethod'] = df_copy['PaymentMethod'].str.replace(' (automatic)', '', regex=False)

        num_cols = df_copy.select_dtypes(include=np.number).columns.tolist()

        for col in num_cols:
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            cutoff = std * 3

            lower, upper = mean - cutoff, mean + cutoff

            outliers = df_copy[(df_copy[col] < lower) | (df_copy[col] > upper)]

            if outliers.empty:
                print(f'No outliers found in {col}')
            else:
                print(f'Outliers found in {col}:')
                print(outliers)

                df_copy = df_copy.drop(outliers.index)

        df_copy_transformed = df_copy.copy()

        label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

        for column in label_encoding_columns:
            if column == 'gender':
                df_copy_transformed[column] = df_copy_transformed[column].map({'Female': 1, 'Male': 0})
            else:
                df_copy_transformed[column] = df_copy_transformed[column].map({'Yes': 1, 'No': 0})

        one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                    'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

        df_copy_transformed = pd.get_dummies(df_copy_transformed, columns = one_hot_encoding_columns)

        min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

        for column in min_max_columns:
            min_column = df_copy_transformed[column].min()
            max_column = df_copy_transformed[column].max()
            df_copy_transformed[column] = (df_copy_transformed[column] - min_column) / (max_column - min_column)
        

        X = df_copy_transformed.drop(columns='Churn')
        Y = df_copy_transformed.loc[:, 'Churn']

        return X, Y
    

    def train_test_split(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=40, shuffle=True)
        return X_train, X_test, Y_train, Y_test
    
    def overSampling(self, X_train, Y_train):
        sm = SMOTE(random_state = 0)
        x_sm, y_sm = sm.fit_resample(X_train, Y_train)
        return x_sm, y_sm
    
    def underSampling(self, X_train, Y_train):
        rus = RandomUnderSampler(random_state = 0)
        x_rus, y_rus = rus.fit_resample(X_train, Y_train)
        return x_rus, y_rus









