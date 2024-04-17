import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def preprocess_data(df):
    # Basic data inspection
    print(df.head())  # Show the first few rows of the DataFrame
    print(df.describe())  # Summary statistics for numerical columns
    print(df.info())  # Info about datatypes and missing values

    # Data cleaning
    # Handling missing values
    
    # Convert data types if necessary
    # df['column_name'] = df['column_name'].astype('int')


    return df
