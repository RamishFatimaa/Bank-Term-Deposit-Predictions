#!/usr/bin/env python
# coding: utf-8

# # Data preparation

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearnex import patch_sklearn
#patch_sklearn()

df0 = pd.read_csv('D:\\Northeastern\\Bank Term Deposit Predictions\\data\\train.csv')
df0.head()


# In[2]:


# Remove Duplicates
df0.drop_duplicates(inplace = True)

# Transfer all unknown in to NaN
df0.replace('unknown', pd.NA, inplace = True)

# Report missing values
df0.isnull().sum()


# In[3]:


# Fill all NaN values with mode
def fillmode(col):
    return col.fillna(col.mode()[0])

df0 = df0.apply(fillmode)


# In[4]:


# Merge month and day, convert as date feature
import calendar
df0['month'] = df0['month'].str.title()
df0['month'] = df0['month'].apply(lambda x: list(calendar.month_abbr).index(x))

df0['date'] = pd.to_datetime(df0['month'].astype(str) + '/' + df0['day'].astype(str), format='%m/%d')
df0['date'] = df0['date'].dt.strftime('%m-%d')


# In[5]:


# Preview dataset again
df0.head()


# # Exploratory data analysis

# In[6]:


# Describe numeric features (outcome = no)
df_n = df0.drop(columns=['month', 'day'])[df0['y'] == 'no']

df_n.describe()


# In[7]:


# Describe numeric features (outcome = yes)
df_y = df0.drop(columns=['month', 'day'])[df0['y'] == 'yes']

df_y.describe()


# In[8]:


# Outcome by Jobs
job_outcome = df0.groupby('job')['y'].value_counts(normalize=True).unstack() * 100
job_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Jobs')
plt.xlabel('Outcome')
plt.ylabel('Job')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()


# In[9]:


# Outcome by marital status
mar_outcome = df0.groupby('marital')['y'].value_counts(normalize=True).unstack() * 100
mar_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Marital Status')
plt.xlabel('Outcome')
plt.ylabel('Marital Status')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()


# In[10]:


# Outcome by education
mar_outcome = df0.groupby('education')['y'].value_counts(normalize=True).unstack() * 100
mar_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Education')
plt.xlabel('Outcome')
plt.ylabel('Education')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()


# In[11]:


# Outcome by default
pd.crosstab(df0['y'], df0['default'])


# In[12]:


# Outcome by housing loan
pd.crosstab(df0['y'], df0['housing'])


# In[13]:


# Outcome by personal loan
pd.crosstab(df0['y'], df0['loan'])


# In[14]:


# Call instances in each month
month_counts = df0['month'].value_counts().sort_index()
month_yes = df0[df0['y'] == 'yes'].groupby('month').size()

plt.figure(figsize=(8, 6))
plt.plot(month_counts.index, month_counts.values, marker='o', linestyle='-', color='blue', label='Total Calls')
plt.plot(month_yes.index, month_yes.values, marker='o', linestyle='-', color='red', label='Successful Calls')
plt.xticks(month_counts.index, [x for x in list(calendar.month_abbr) if x != ''])
plt.title('Call Instances in Each Month')
plt.xlabel('Month')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()
plt.show()


# In[15]:


# Durations of call and successful calls
sns.boxplot(x='y', y='duration', data=df0)
plt.title('Boxplot of Call Duration by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Call Duration (minutes)')
plt.tight_layout()
plt.show()

mean_duration = df0.groupby('y')['duration'].mean()
print(mean_duration)


# # Data modelling

# In[16]:


# Feature encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
df = df0

# Remove time series features
df0.drop(columns=['month', 'day', 'date'], inplace = True)

# Binary features
df['default'] = pd.get_dummies(df['default'])['yes']
df['housing'] = pd.get_dummies(df['housing'])['yes']
df['loan'] = pd.get_dummies(df['loan'])['yes']
df['y'] = pd.get_dummies(df['y'])['yes']
df.replace({True: 1, False: 0}, inplace = True)

# Ordered categorical features
df['education'] = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']]).fit_transform(df[['education']])
df['poutcome'] = OrdinalEncoder(categories=[['failure', 'success', 'other']]).fit_transform(df[['poutcome']])

# Inordered catrgorical features (one-hot encoding)
df['marital'] = OneHotEncoder(sparse_output = False).fit_transform(df[['marital']])
df['contact'] = OneHotEncoder(sparse_output = False).fit_transform(df[['contact']])
df['marital'] = OneHotEncoder(sparse_output = False).fit_transform(df[['marital']])

# Inordered catrgorical features (target encoding)
target_code = df.groupby('job')['y'].mean()
df['job'] = df['job'].map(target_code)


# In[17]:


# Split train and test set
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)


# In[18]:


# Decision tree modelling
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 2024)
dt.fit(X_train, y_train)

# Make prediction on test set
y_pred = dt.predict(X_test)


# In[19]:


# Model visualization
from sklearn.tree import export_graphviz
import graphviz
import subprocess

export_graphviz(dt, out_file="dt.dot", feature_names=X.columns.astype(str).tolist(), class_names=y.unique().astype(str).tolist(),
                filled=True, rounded=True, special_characters=True)

with open("dt.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[20]:


# Performance report on test set
from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_test, y_pred))


# In[21]:


# Confusion matrix
from sklearn.metrics import confusion_matrix

test_result = dt.predict(X_test)
matrix = confusion_matrix(y_test, test_result)

print(matrix)


# In[22]:


# ROC curve
from sklearn.metrics import roc_curve, auc

y_score = dt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='cyan', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='magenta', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print('AUC:', roc_auc)


# In[23]:


# Cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
scores

