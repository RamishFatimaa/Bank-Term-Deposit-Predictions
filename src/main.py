# main.py

# Import necessary modules
import os
import sys
import pandas as pd

scripts_dir = 'D:\\Northeastern\\Bank Term Deposit Predictions\\scripts'
sys.path.insert(1, scripts_dir)
import data_preprocessing  # Imports the preprocessing module
import Question2
import Question3
def main():
    df_path = 'D:\\Northeastern\\Bank Term Deposit Predictions\\data\\dataset.csv'
    try:
        df = pd.read_csv(df_path)
        print("Data Loaded Successfully")
        print("Initial DataFrame head:")
        print(df.head())
    except Exception as e:
        print("Failed to load data:", e)
        return

    # Preprocess the data with a single function call
    cleaned_df = data_preprocessing.preprocess_data(df)
    print("Data preprocessing completed")
    print("Cleaned DataFrame head:")
    print(cleaned_df.head())

if __name__ == "__main__":
    main()
