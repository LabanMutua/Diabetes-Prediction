# Diabetes Prediction

## Overview
This project aims to build a predictive model to determine whether a patient has diabetes based on diagnostic measurements. The dataset used for this analysis is sourced from Kaggle, and it includes several health-related features for female patients of Pima Indian heritage.

The key objective is to identify the most suitable machine learning model for predicting diabetes, considering factors like model accuracy, performance on class imbalance, and generalization ability.

## Data Source
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

## Key Features
- **Data Inspection**: Checks for missing values, duplicates, and ensures proper data types.
- **Exploratory Data Analysis**: Includes visualizations of the distribution of variables, correlations, and insights on class imbalance.
- **Modeling**: Several machine learning models were tested, including Logistic Regression, Decision Trees, K-Nearest Neighbors, Bagging Classifier, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Support Vector Machine.
- **Model Evaluation**: Evaluates performance based on testing accuracy, precision, recall, and F1-score for both classes.

## Project Workflow
All project work is available on [Diabetes Prediction.ipynb](https://github.com/LabanMutua/Diabetes-Prediction/blob/main/Diabetes%20Prediction.ipynb). It consists of:
1. **Data Preparation**
   - Cleaned the dataset by handling outliers and imputing missing values where necessary.
   - Split the data into training and testing sets.

2. **Exploratory Data Analysis (EDA)**
   - Conducted a thorough analysis of the data distribution and correlations.
   - Analyzed the class imbalance and handled outliers accordingly.

3. **Model Building**
   - Several machine learning algorithms were tested and evaluated on their ability to predict diabetes.
   - Hyperparameter tuning was applied to optimize model performance.

4. **Model Comparison**
   - **Best Model**: Random Forest (Tuned Parameters) achieved the highest testing accuracy of 0.77 with balanced precision and recall.
   - **Alternative Model**: Bagging Classifier with testing accuracy of 0.75.
   - Models such as Gradient Boosting and XGBoost were also considered due to their strong predictive power but showed marginal improvements after tuning.

## Conclusion
- **Adopted Model**: Random Forest with tuned parameters is the best model due to its high accuracy, balanced performance, and robust generalization.
- **Alternative Models**: Bagging Classifier and Gradient Boosting are also strong contenders for cases requiring different performance characteristics.

