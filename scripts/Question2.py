#!/usr/bin/env python
# coding: utf-8

# #### Importing the dataset

# In[110]:





# In[92]:
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\dataset.csv")


# In[93]:


df.head()


# In[65]:


df.shape


# In[7]:


df.dtypes


# In[120]:


train=pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\train.csv")


# In[122]:


train.shape


# In[136]:


test=pd.read_csv("D:\\Northeastern\\Bank Term Deposit Predictions\\data\\test.csv")


# In[138]:


test.shape


# #### Data preprocessing

# In[13]:


# Check for missing values
missing_values = train.isnull().sum()
print("Missing values:\n", missing_values)

# Explore the dataset
print("\nDataset information:")
print(train.info())


# In[123]:


# Check for duplicate rows
duplicate_rows = train.duplicated()

# Count the number of duplicate rows
num_duplicates = duplicate_rows.sum()

# Print the number of duplicate rows
print("Number of duplicate rows:", num_duplicates)


# In[124]:


train['education'].unique()


# In[125]:


# Replace 'unknown' values with NaN
train['education'] = train['education'].replace('unknown', np.nan)

# Impute NaN values with the most frequent category, which is 'secondary'
most_frequent_education = 'secondary'
train['education'] = train['education'].fillna(most_frequent_education)


# In[148]:


# Replace 'unknown' values with NaN
test['education'] = test['education'].replace('unknown', np.nan)

# Impute NaN values with the most frequent category, which is 'secondary'
most_frequent_education = 'secondary'
test['education'] = test['education'].fillna(most_frequent_education)


# #### EDA

# In[126]:


# Calculate summary statistics for numerical variables
summary_stats = train.describe()

# Transpose the summary statistics for better readability
summary_stats = summary_stats.T

# Print the summary statistics
print("Summary Statistics for Numerical Variables:")
print(summary_stats)


# In[127]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical columns
numerical_cols = train.select_dtypes(include=['int64']).columns

# Plot box plots for each numerical variable
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_cols])
plt.title('Boxplot of Numerical Variables')
plt.xticks(rotation=45)
plt.show()


# In[128]:


# Plot box plot for 'balance'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['balance'])
plt.title('Boxplot of Balance')
plt.xlabel('Balance')
plt.show()


# #### Distributions of Numerical Variables

# In[129]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and axes
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Plot histograms for numerical variables
for i, column in enumerate(train.select_dtypes(include=['int64']).columns):
    sns.histplot(df[column], ax=axes[i], kde=True, bins=30, color='skyblue')
    axes[i].set_title(column)

# Adjust layout
plt.tight_layout()
plt.show()


# #### Relationship with Target Variable

# #### Count Plots for Categorical Variables

# In[131]:


# Set up the figure and axes
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Plot count plots for categorical variables
for i, column in enumerate(train.select_dtypes(include=['object']).columns):
    sns.countplot(y=column, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(column)

# Adjust layout
plt.tight_layout()
plt.show()


# In[132]:


# Encode the target variable 'y' into numeric format
train['y_encoded'] = train['y'].map({'no': 0, 'yes': 1})

# Drop non-numeric columns
numeric_df = train.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Get the correlation of each feature with the encoded target variable 'y_encoded'
correlation_with_target = correlation_matrix['y_encoded'].sort_values(ascending=False)

# Display the top correlated features
print(correlation_with_target)


# In[143]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable 'y' in the test set
test['y_encoded'] = label_encoder.fit_transform(test['y'])




# In[133]:


import seaborn as sns
import matplotlib.pyplot as plt

# Reshape correlation_with_target into a DataFrame
correlation_df = correlation_with_target.to_frame()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix with Target Variable')
plt.show()


# In[134]:


# Plot relationship with target variable
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

sns.countplot(x='y', data=train, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Count of Subscriptions')

sns.boxplot(x='y', y='age', data=train, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Age vs Subscriptions')

sns.boxplot(x='y', y='balance', data=train, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Balance vs Subscriptions')

sns.boxplot(x='y', y='duration', data=train, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Duration vs Subscriptions')

# Adjust layout
plt.tight_layout()
plt.show()


# In[135]:


print(train.columns)


# ##### Model

# In[152]:


#One-hot encode categorical variables for train set
X_train = pd.get_dummies(train.drop(columns=['y']))
y_train = train['y']

# One-hot encode categorical variables for test set
X_test = pd.get_dummies(test.drop(columns=['y']))
y_test = test['y']

# Initialize and train the logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print coefficients
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_[0]})
print(coefficients)

print("Train set shape:", train.shape)
print("Test set shape:", test.shape)


# In[153]:


# Compute precision
precision = precision_score(y_test, y_pred, pos_label='yes')

# Compute recall
recall = recall_score(y_test, y_pred, pos_label='yes')

# Compute F1-score
f1 = f1_score(y_test, y_pred, pos_label='yes')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Compute probabilities for ROC curve
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label='yes')
auc = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[158]:


# Perform k-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Use cv=5 for 5-fold cross-validation

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate the mean and standard deviation of the cross-validation scores
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

# In[ ]:




