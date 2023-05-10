# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:03:57 2023

@author: alan
"""
#https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4
#https://towardsdatascience.com/breaking-the-curse-of-small-datasets-in-machine-learning-part-1-36f28b0c044d
#https://www.kaggle.com/code/rafjaa/dealing-with-very-small-datasets
#https://towardsdatascience.com/getting-deeper-into-categorical-encodings-for-machine-learning-2312acd347c8
#https://towardsdatascience.com/4-categorical-encoding-concepts-to-know-for-data-scientists-e144851c6383


import category_encoders as ce
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import *


import pandas as pd;
data = [['Salt Lake City', 10, 120], ['Seattle', 5, 120], ['San Franscisco', 5, 140], 
        ['Seattle', 3, 100], ['Seattle', 1, 70], ['San Franscisco', 2, 100],['Salt Lake City', 1, 60], 
        ['San Franscisco', 2, 110], ['Seattle', 4, 100],['Salt Lake City', 2, 70] ]
df = pd.DataFrame(data, columns = ['City', 'Years OF Exp','Yearly Salary in Thousands'])




#encoding
tenc=ce.leave_one_out.LeaveOneOutEncoder(verbose=1,sigma=0.05)
df_city=tenc.fit_transform(df['City'],df['Yearly Salary in Thousands'])
df['City']=df_city

#%% Expansion of the data set
## Multiply every item by -1
df_neg = df.mul(-1)
#expand the original dataframe
df_expanded = pd.concat([df, df_neg], axis=0)

#%%get X_train, X_test, y_train, y_test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
target='Yearly Salary in Thousands'
y_train=df_expanded[target]
X_train=df_expanded.drop(columns=([target]))

#numeric_dataset = tenc.transform(df['City'])


#build a ML model
clf = DecisionTreeClassifier(max_depth=1)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
df_expanded['prediction']=y_train_pred



#Linear - 
clf=Ridge()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
df_expanded['prediction']=y_train_pred

'''

df1=pd.read_csv('Book2.csv')
y=df1['y']
X=df1.drop(columns=['y'])
enc = TargetEncoder(cols=[ 'CentralAir',  'Heating'], min_samples_leaf=20, smoothing=10).fit(X, y)
numeric_dataset = enc.transform(X)
numeric_dataset['y']=y
numeric_dataset.to_clipboard()


display_cols = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'YearBuilt', 'Heating', 'CentralAir']
bunch = fetch_openml(name='house_prices', as_frame=True)
y = bunch.target > 200000
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
enc = TargetEncoder(cols=[ 'CentralAir',  'Heating'], min_samples_leaf=20, smoothing=10).fit(X, y)
numeric_dataset = enc.transform(X)
numeric_dataset['y']=y
print(numeric_dataset.info())
'''
print('finish')
