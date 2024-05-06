#!/usr/bin/env python
# coding: utf-8

# # Data preparation

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearnex import patch_sklearn
# patch_sklearn()

df0 = pd.read_csv('train.csv')
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


df0.drop(columns=['month', 'day']).describe()


# In[9]:


# Outcome by Jobs
job_outcome = df0.groupby('job')['y'].value_counts(normalize=True).unstack() * 100
a1 = job_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Jobs')
plt.xlabel('Outcome')
plt.ylabel('Job')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
job_outcome


# In[10]:


# Outcome by marital status
mar_outcome = df0.groupby('marital')['y'].value_counts(normalize=True).unstack() * 100
mar_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Marital Status')
plt.xlabel('Outcome')
plt.ylabel('Marital Status')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
mar_outcome


# In[11]:


# Outcome by education
edu_outcome = df0.groupby('education')['y'].value_counts(normalize=True).unstack() * 100
edu_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Education')
plt.xlabel('Outcome')
plt.ylabel('Education')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
edu_outcome


# In[12]:


# Outcome by means of contact
con_outcome = df0.groupby('contact')['y'].value_counts(normalize=True).unstack() * 100
con_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Means of Contact')
plt.xlabel('Outcome')
plt.ylabel('Contact')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
con_outcome


# In[13]:


# Outcome by default
def_outcome = df0.groupby('default')['y'].value_counts(normalize=True).unstack() * 100
def_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Default')
plt.xlabel('Outcome')
plt.ylabel('Default')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
def_outcome


# In[14]:


# Outcome by housing loan
house_outcome = df0.groupby('housing')['y'].value_counts(normalize=True).unstack() * 100
house_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Housing Loan')
plt.xlabel('Outcome')
plt.ylabel('Housing')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
house_outcome


# In[15]:


# Outcome by personal loan
loan_outcome = df0.groupby('loan')['y'].value_counts(normalize=True).unstack() * 100
loan_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Personal Loan')
plt.xlabel('Outcome')
plt.ylabel('Loan')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
loan_outcome


# In[16]:


# Outcome by outcome of previous campaign
pre_outcome = df0.groupby('poutcome')['y'].value_counts(normalize=True).unstack() * 100
pre_outcome.plot(kind='barh', stacked=True)
plt.title('Outcome by Outcome of Previous Campaign')
plt.xlabel('Outcome')
plt.ylabel('Previous')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()
pre_outcome


# In[17]:


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


# In[18]:


# Durations of call and successful calls
sns.boxplot(x='y', y='duration', data=df0)
plt.title('Boxplot of Call Duration by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Call Duration (minutes)')
plt.tight_layout()
plt.show()

mean_duration = df0.groupby('y')['duration'].mean()
mean_duration


# # Data modelling

# Feature Encoding

# In[19]:


# Feature encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
df = df0

# Remove time series features
df0.drop(columns=['month', 'day', 'date'], inplace = True)

# Remove unrelative features
df0.drop(columns=['marital', 'contact'], inplace = True)

# Binary features
df['default'] = pd.get_dummies(df['default'])['yes']
df['housing'] = pd.get_dummies(df['housing'])['yes']
df['loan'] = pd.get_dummies(df['loan'])['yes']
df['y'] = pd.get_dummies(df['y'])['yes']
df.replace({True: 1, False: 0}, inplace = True)

# Ordered categorical features
df['poutcome'] = OrdinalEncoder(categories=[['failure', 'success', 'other']]).fit_transform(df[['poutcome']])
df['education'] = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']]).fit_transform(df[['education']])

# Unordered catrgorical features (one-hot encoding)
df['job'] = OneHotEncoder(sparse_output = False).fit_transform(df[['job']])

# Unused Target Encoding
#target_code = df.groupby('job')['y'].mean()
#df['job'] = df['job'].map(target_code)


# In[20]:


# Split train and test set
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)


# Decision Tree

# In[55]:


# Searching for optimal parameter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt_cl = DecisionTreeClassifier()
param_dt = {'max_depth': list(range(1, 11)),
    'min_samples_split': list(range(1, 6)),
    'min_samples_leaf': list(range(1, 6))}

grid_dt = GridSearchCV(estimator = dt_cl, param_grid = param_dt, cv = 10)
grid_dt.fit(X_train, y_train)

print("Optimal Parameters:", grid_dt.best_params_)
print("Best Score:", grid_dt.best_score_)


# In[56]:


# Decision tree modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(max_depth = 5, min_samples_split = 2, min_samples_leaf = 1, random_state = 2024)
dt.fit(X_train, y_train)

# Make prediction on test set
y_pred_dt = dt.predict(X_test)


# In[22]:


# Model visualization
from sklearn.tree import export_graphviz
import graphviz
import subprocess

export_graphviz(dt, out_file="dt.dot", feature_names=X.columns.astype(str).tolist(), class_names=y.unique().astype(str).tolist(),
                filled=True, rounded=True, special_characters=True)

with open("dt.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[23]:


# Performance report on test set
from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_test, y_pred_dt))


# In[24]:


# Confusion matrix
from sklearn.metrics import confusion_matrix

matrix_dt = confusion_matrix(y_test, y_pred_dt)
print(matrix_dt)


# In[25]:


# ROC curve
from sklearn.metrics import roc_curve, auc

y_score_dt = dt.predict_proba(X_test)[:, 1]
fpr1, tpr1, thre1 = roc_curve(y_test, y_score_dt)
roc_auc1 = auc(fpr1, tpr1)

plt.figure()
plt.plot(fpr1, tpr1, color='cyan', lw=2, label='ROC curve (area = %0.2f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='magenta', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

bthre1 = thre1[np.argmax(tpr1 - fpr1)]
print('Best Threshold:', bthre1)


# In[26]:


# Confusion matrix by revised threshold
y_dt_adjusted = (y_score_dt >= bthre1).astype(int)
matrix_dt2 = confusion_matrix(y_test, y_dt_adjusted)

print(matrix_dt2)


# In[27]:


# Cross validation
from sklearn.model_selection import cross_val_score

scores_dt = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
scores_dt


# AdaBoost

# In[66]:


# AdaBoost modelling
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(estimator = dt, n_estimators = 100, algorithm = 'SAMME', random_state = 2024)
ada.fit(X_train, y_train)

# Make prediction on test set
y_pred_ada = ada.predict(X_test)


# In[67]:


# Performance report on test set
print(classification_report(y_test, y_pred_ada))


# In[68]:


# Confusion matrix
matrix_ada = confusion_matrix(y_test, y_pred_ada)
print(matrix_ada)


# In[69]:


# ROC curve
y_score_ada = ada.predict_proba(X_test)[:, 1]
fpr2, tpr2, thre2 = roc_curve(y_test, y_score_ada)
roc_auc2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr2, tpr2, color='cyan', lw=2, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='magenta', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

bthre2 = thre2[np.argmax(tpr2 - fpr2)]
print('Best Threshold:', bthre2)


# In[70]:


# Confusion matrix by revised threshold
y_ada_adjusted = (y_score_ada >= bthre2).astype(int)
matrix_ada2 = confusion_matrix(y_test, y_ada_adjusted)

print(matrix_ada2)


# In[71]:


# Cross validation
scores_ada = cross_val_score(ada, X, y, cv=10, scoring='accuracy')
scores_ada

