# Import necessary modules
from data_preprocessing import preprocess_data
import pandas as pd

def main():
    df_path = 'D:\\Northeastern\\Bank Term Deposit Predictions\\dataset.csv'
    # Read the data into DataFrame
    df = pd.read_csv(df_path)
    print("Data Loaded Successfully")
    
    # Preprocess the data
    cleaned_df = preprocess_data(df)
    print("Data preprocessing completed")
    print("Test")
    # You can now proceed with cleaned_df for further analysis or modeling

if __name__ == "__main__":
    main()
