# Bank Term Deposit Predictions Dataset

## Group Members
- Nidhi
- Jeetesh
- Ramish
- Fatima
- Yukang Lin

## Overview
This project focuses on predicting customer subscription to term deposits based on historical marketing campaign data. By leveraging machine learning techniques, we aim to identify potential subscribers and optimize marketing efforts.

## Dataset

## Data Source
This dataset is sourced from [Kaggle: Bank Term Deposit Predictions](https://www.kaggle.com).

### **Description:**
The dataset contains records of a bank's marketing campaign interactions with customers. 

- **Training set:** 45,211 instances
- **Test set:** 4,521 instances
- **Features:** Age, job, marital status, education, balance, contact type, previous campaign outcomes, etc.
- **Target Variable:** `y` (Indicates whether the customer subscribed to a term deposit: `yes` or `no`)

## Dataset Description

Below is a description of each field in the dataset:

| Field     | Description                                                                 | Type        |
|-----------|-----------------------------------------------------------------------------|-------------|
| `age`     | The age of the customer.                                                    | Numerical   |
| `job`     | The occupation/employment status of the customer.                           | Categorical |
| `marital` | The marital status of the customer.                                         | Categorical |
| `education` | The education level attained by the customer.                              | Categorical |
| `default` | Whether the customer has credit in default.                                 | Categorical |
| `balance` | The balance in the customer's account.                                      | Numerical   |
| `housing` | Whether the customer has a housing loan.                                    | Categorical |
| `contact` | Type of communication used to contact customers (phone, cellular, etc.).    | Categorical |
| `day`     | Day of the month when customers were last contacted.                        | Numerical   |
| `duration` | Duration (in seconds) of the last contact with customers during the previous campaign. | Numerical |
| `pdays`   | The number of days passed by after contact from the previous campaign.      | Numerical   |
| `poutcome`| Outcome from the previous marketing campaign.                               | Categorical |


## Business Questions
1. **What factors influence a customer's decision to invest in a term deposit?**
2. **Can the bank's operational status be inferred from customer usage records?**
3. **Can we predict term deposit subscription likelihood and optimize marketing efforts?**
4. **Can we classify customers into potential vs. non-potential subscribers?**

## Challenges
- **Class imbalance:** More negative instances (`no`) than positive (`yes`)
- **Handling missing values and categorical variables**
- **Feature selection and importance analysis**

## Data Preprocessing
### **Handling Missing Values:**
- No missing values detected
- Replaced `unknown` values in `education` column with the most frequent value (`secondary`)
- Merged `month` and `day` columns to create a proper date format

## Exploratory Data Analysis (EDA)
### **Findings:**
- Customers with **higher balances** and **longer call durations** were more likely to subscribe.
- **Students and retirees** showed higher subscription rates.
- **Higher education levels** correlated with increased subscriptions.
- Clients with **no default records, no housing, or personal loans** had a greater likelihood of subscribing.
- The **month of contact** significantly affected the subscription rate.

## Model Development
### **Machine Learning Models Used:**
1. **Random Forest Classifier**
   - Provided baseline feature importance.
   - Identified balance, duration, and contact type as key features.
2. **Neural Networks**
   - Captured complex, non-linear interactions between features.
   - Identified education and job type as additional significant predictors.
3. **Logistic Regression**
   - Achieved **89.88% accuracy** after feature selection.
   - Used for baseline comparisons.
4. **XGBoost Classifier**
   - Achieved **90.55% accuracy** after hyperparameter tuning.
   - Strong feature importance insights.
5. **Decision Tree**
   - Tuned via GridSearchCV (optimal depth: 5 layers).
   - Confusion matrix and ROC-AUC analysis.
6. **Adaptive Boosting (AdaBoost)**
   - Improved classification balance.
   - Increased True Positive Rate while controlling False Negatives.

## Feature Importance Analysis
- **Key Features:**
  - `balance`: Higher balance → Higher subscription likelihood
  - `duration`: Longer call duration → Increased success rate
  - `month`: Certain months (Aug, Sep) showed higher engagement
  - `job & education`: Higher education and job types mattered
  - `previous campaign outcome`: Customers with previous successful interactions were more likely to subscribe

## Model Performance Comparison
| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression | 89.88%   |
| XGBoost           | 90.55%   |
| Decision Tree     | 89.01%   |
| AdaBoost         | Improved TPR & FPR |

## Insights & Recommendations
- **Targeted Marketing:** Focus on **high-balance** customers with **longer call engagement**.
- **Optimal Timing:** Execute campaigns in **August and September** for maximum conversion.
- **Customer Segmentation:** Classify customers based on **past interaction success rates**.
- **Campaign Optimization:** Reduce excessive follow-ups to avoid customer fatigue.

## Conclusion
- **Predictive modeling improves marketing efficiency** by identifying potential subscribers.
- **Feature engineering plays a crucial role** in enhancing model accuracy.
- **Advanced techniques like XGBoost and AdaBoost** offer superior classification performance.

## Future Enhancements
- **Further Hyperparameter Tuning** for improved model performance.
- **Customer Profiling & Segmentation** using clustering techniques.
- **A/B Testing for Campaign Strategies** to refine marketing effectiveness.

## Thank You! 🚀
