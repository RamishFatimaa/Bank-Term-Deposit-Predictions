# main.py

# Import necessary modules
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
scripts_dir = 'D:\\Northeastern\\Bank Term Deposit Predictions\\scripts'
sys.path.insert(1, scripts_dir)
#import data_preprocessing  # Imports the preprocessing module
import Question1
#import Question2
#import Question3

def main():
    try:
        # Load your dataset
        train = pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\train.csv")
        print("Data Loaded Successfully")
        print("Initial DataFrame head:")
        print(train.head())

        # Run multiple models on the dataset
        num_features = train.drop('y', axis=1).shape[1]  # Determine the number of features based on the dataset
        
        # Correctly passing the number of features to the lambda function
        Question1.run_multiple_models(train, lambda num_features=num_features: Question1.create_models(num_features))

    except Exception as e:
        print("Failed to load data:", e)

if __name__ == "__main__":
    main()