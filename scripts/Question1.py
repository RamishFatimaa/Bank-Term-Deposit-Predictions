import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


def preprocess_data(df):
    print("Starting preprocessing...")
    binary_columns = ['default', 'housing', 'loan', 'y']
    for column in binary_columns:
        df[column] = df[column].map({'yes': 1, 'no': 0})
    return df

def load_and_preprocess_data(df, train, test):
    # Analyze train data before preprocessing
    analyze_data(train, 'Train Data Analysis')
    
    # Preprocess the full dataset
    cleaned_df = preprocess_data(df)
    print("Data preprocessing completed for full dataset")

    # Preprocess train and test datasets
    cleaned_train = preprocess_data(train)
    cleaned_test = preprocess_data(test)
    print("Data preprocessing completed for train and test datasets")

    return cleaned_df, cleaned_train, cleaned_test

import pandas as pd

def analyze_data(df, title):
    print(f"\n{title}")
    print("Summary statistics:")
    print(df.describe(include='all'))

    # Converting binary columns from 'yes'/'no' to 1/0
    binary_columns = ['default', 'housing', 'loan', 'y']
    for column in binary_columns:
        df[column] = df[column].map({'yes': 1, 'no': 0}).astype(float)

    # One-hot encode categorical variables for correlation analysis
    categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Checking correlation with the target variable 'y' if it's binary
    if 'y' in df.columns and df['y'].dtype != object:
        print("Correlation with the target variable 'y':")
        print(df.corr()['y'].sort_values())

    # Full correlation matrix
    print("Correlation matrix:")
    print(df.corr())
    
    # Full correlation matrix
    print("Full correlation matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    # Plotting correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")
    #plt.show()
    
    importances, X_train, X_test, y_train, y_test = feature_importance_analysis(df)
    print("Feature Importances:\n", importances)

    # Select only the top N features for the final model
    top_features = importances.head(10).index.tolist()  # Adjust N based on your preference
    train_final_model(X_train, y_train, X_test, y_test, top_features)

# Function to perform feature importance analysis using Random Forest
def feature_importance_analysis(df):
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    return importances, X_train, X_test, y_train, y_test

# Function to train and evaluate the final model
def train_final_model(X_train, y_train, X_test, y_test, top_features):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[top_features], y_train)
    
    predictions = model.predict(X_test[top_features])
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    
    # Plot ROC curve
    y_score = model.predict_proba(X_test[top_features])[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

