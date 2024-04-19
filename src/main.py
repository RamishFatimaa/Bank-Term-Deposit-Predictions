# Import necessary modules
import sys
sys.path.append('scripts')  # Adds the 'scripts' directory to the path
import data_preprocessing  # Imports the preprocessing module

import pandas as pd

def main():
    df_path = 'D:\\Northeastern\\Bank Term Deposit Predictions\\data\\dataset.csv'
    # Read the data into DataFrame
    df = pd.read_csv(df_path)
    print("Data Loaded Successfully")
    
    # Preprocess the data
    #cleaned_df = preprocess_data(df)
    print("Data preprocessing completed")
    print("Test")
    print("Hi")
   

if __name__ == "__main__":
    main()
