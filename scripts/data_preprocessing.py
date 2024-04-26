# data_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    print("Starting preprocessing...")
    print("Initial DataFrame columns:", df.columns.tolist())
    
    df = basic_data_inspection(df)
    if df is None:
        print("Error after basic_data_inspection")
        return None
    
    df = handle_missing_values(df)
    if df is None:
        print("Error after handle_missing_values")
        return None
    
    
    important_columns = ['balance', 'age', 'duration', 'campaign', 'pdays']
    df = detect_outliers_isolation_forest(df, important_columns)
    #df = normalize_data(df, ['balance', 'age', 'duration', 'campaign', 'pdays'])
    
    
    
    return df

def basic_data_inspection(df):
    print("Performing basic data inspection...")
    print(df.head())
    print(df.describe())
    df.shape
    print("Data types and missing values info:")
    info = df.info()
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated()
    # Count the number of duplicate rows
    num_duplicates = duplicate_rows.sum()

    # Print the number of duplicate rows
    print("Number of duplicate rows:", num_duplicates)
    
    unique_education_values = df['education'].unique()
    print("Number of unique values in Education:",unique_education_values)
    
    # Replace 'unknown' values with NaN
    df['education'] = df['education'].replace('unknown', np.nan)

    # Impute NaN values with the most frequent category, which is 'secondary'
    most_frequent_education = 'secondary'
    df['education'] = df['education'].fillna(most_frequent_education)

    if 'object' in df.dtypes.values:
        print("Categorical features description:")
        print(df.select_dtypes(include=['object']).describe())
    else:
        print("No categorical (object type) columns to describe.")
        
    # Visualization of distributions of numerical features
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        #plt.show()
        
    sns.pairplot(df.select_dtypes(include=[np.number]), diag_kind='kde')
    #plt.show()
    
    pd.plotting.scatter_matrix(df[['balance', 'age', 'campaign', 'pdays']], figsize=(10, 10))
    #plt.show()

    return df

def handle_missing_values(df):
    print("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in {col}.")
    return df

def detect_outliers_isolation_forest(df, columns):
    for column_name in columns:
        print(f"Detecting outliers in {column_name}...")
        
        if column_name not in df.columns:
            print(f"Error: {column_name} is not in DataFrame.")
            continue  # Skip this column and continue with the next

        try:
            # Applying Isolation Forest to detect outliers
            clf = IsolationForest(random_state=42, contamination='auto')
            clf.fit(df[[column_name]])  # Fit model on the column
            is_outlier = clf.predict(df[[column_name]])
            outlier_column = 'is_outlier_' + column_name
            df[outlier_column] = (is_outlier == -1).astype(int)

            # Printing out detected outliers
            outliers = df[df[outlier_column] == 1]
            if not outliers.empty:
                print("Detected Outliers:")
                print(outliers[[column_name, 'job', 'marital', 'education', 'campaign']])
        except Exception as e:
            print(f"An error occurred while processing {column_name}: {e}")

    return df
        
def recode_pdays(df):
    # Check if pdays column exists to avoid runtime errors
    if 'pdays' not in df.columns:
        print("Error: pdays column is not in DataFrame.")
        return df

    # Re-coding pdays directly to handle no previous contact and numeric conversion
    df['pdays_recode'] = df['pdays'].apply(lambda x: 'no_previous_contact' if x == -1 else x)

    # Define bins and labels for pdays that are numeric
    bins = [0, 7, 14, 21, 28, 60, 120, 180, 871, np.inf]
    labels = ['within_1_week', 'within_2_weeks', 'within_3_weeks', 
              'within_4_weeks', 'within_2_months', 'within_4_months', 
              'within_6_months', 'more_than_6_months', 'never_contacted']

    # Create a temporary column for numeric pdays values, setting -1 to NaN for now
    df['pdays_numeric'] = df['pdays'].replace(-1, np.nan)

    # Apply binning to numeric values of pdays
    df['pdays_numeric'] = pd.cut(df['pdays_numeric'], bins=[-np.inf] + bins, labels=['no_previous_contact'] + labels, right=False)

    # Ensure all original -1 values are categorized correctly
    df['pdays_recode'] = np.where(df['pdays'] == -1, 'no_previous_contact', df['pdays_numeric'])

    return df

from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, columns):
    """Normalize specified numeric columns using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

