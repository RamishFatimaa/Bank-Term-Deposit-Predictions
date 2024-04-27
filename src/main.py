# main.py

# Import necessary modules
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

scripts_dir = 'D:\\Northeastern\\Bank Term Deposit Predictions\\scripts'
sys.path.insert(1, scripts_dir)
#import data_preprocessing  # Imports the preprocessing module
import Question1
import Question2
import Question3

def main():
    df_path = 'D:\\Northeastern\\Bank Term Deposit Predictions\\data\\dataset.csv'
    try:
        df = pd.read_csv(df_path)
        print(df.head())
        train=pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\train.csv")
        test=pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\test.csv")
        print("Data Loaded Successfully")
        print("Initial DataFrame head:")
    except Exception as e:
        print("Failed to load data:", e)
        return

    # Preprocess the data with a single function call
    #cleaned_df = data_preprocessing.preprocess_data(df)
    #print("Data preprocessing completed")
    #print("Cleaned DataFrame head:")
    #print(cleaned_df.head())
    
    # Preprocess the test data
    cleaned_df, cleaned_train, cleaned_test = Question1.load_and_preprocess_data(df, train, test)
    print("Data preprocessing completed")
    print("Cleaned Train Data head:")
    print(cleaned_train.head())
    
    
if __name__ == "__main__":
    main()
