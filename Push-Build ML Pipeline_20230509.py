# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:54:14 2023

@author: alan


Want to achieve the following goals:
    - compare multiple ML method
    - enabled gridsearch in each method
    - deal with small sample

"""


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, mean_squared_error, r2_score,accuracy_score

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#df = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])


# Load the iris dataset
iris = load_iris()

# Create a dataframe from the iris data
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
y = iris.target

# Define the pipelines
pipelines = {
    'label_encoding_logistic_regression': Pipeline(steps=[
        ('label_encoder', LabelEncoder()),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),
    'one_hot_encoding_logistic_regression': Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(drop='first')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),    
    'one_hot_encoding_decision_tree': Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(drop='first')),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier())
    ])
}

pipelines = {
    'one_hot_encoding_decision_tree': Pipeline(steps=[
        ('classifier', DecisionTreeClassifier())
    ])
}



# Define the hyperparameter grids for each classifier
param_grids = {
    'label_encoding_logistic_regression': {
        'classifier__C': [0.1, 1, 10]
    },
    'one_hot_encoding_logistic_regression': {
        'classifier__C': [0.1, 1, 10]
    },
    'one_hot_encoding_decision_tree': {
        'classifier__max_depth': [1],
        'classifier__min_samples_leaf': [1]
    }
}

# Define the RMSE scoring function
#rmse_scorer = make_scorer(mean_squared_error, squared=False,greater_is_better=False)



# Fit and evaluate the pipelines using leave-one-out cross-validation and hyperparameter grid search
loo = LeaveOneOut()
for pipeline_name, pipeline in pipelines.items():
    param_grid = param_grids[pipeline_name]
    grid_search = GridSearchCV(pipeline, param_grid,scoring='accuracy', cv=loo,return_train_score=True)
    grid_search.fit(X, y)
    
    # Print the best hyperparameters and the corresponding RMSE score for the random forest regression model
    print(f"Best hyperparameters for  {pipeline_name}: ", grid_search.best_params_)
    print(f"Best RMSE score for {pipeline_name}:",grid_search.best_score_)
    
    # Calculate the training accuracy, https://www.kaggle.com/getting-started/27261
    train_accuracy = grid_search.best_estimator_.score(X, y)
    
    # Calculate the LOOCV accuracy
    loocv_accuracy = grid_search.best_score_
    
    # Check for overfitting
    if train_accuracy - loocv_accuracy > 0.1:
        print(f'Warning: {pipeline_name} is overfitting (training accuracy = {train_accuracy}, LOOCV accuracy = {loocv_accuracy})')
    else:
        print(f'{pipeline_name}: training accuracy = {train_accuracy}, LOOCV accuracy = {loocv_accuracy}')
