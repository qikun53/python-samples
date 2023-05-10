# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:54:47 2023

@author: alan
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
#import pydotplus
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image
import os
import category_encoders as ce
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
#https://stackoverflow.com/questions/57488294/how-to-download-the-image-created-using-graphviz


#read data
data = pd.read_csv("202305 Feature Importance.csv")
data.info()
y = data['play']
X = data.drop("play", axis=1)
columns = X.columns

encoder = ce.OrdinalEncoder(data.columns)

X = encoder.fit_transform(X)

#criterion='gini'
criterion='entropy'

clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf.fit(X, y)

#plt.figure(figsize=(12,8))
from sklearn import tree
#tree.plot_tree(clf.fit(X, y)) 


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                              feature_names=X.columns,  
                              class_names=y,  
                              filled=True, rounded=True,  
                              special_characters=True)


'''

# Replace encoded values with original values in the text representation
for col, mapping in mapping_dict.items():
    for key, value in mapping.items():
        text_representation = text_representation.replace(f"{col} <= {value:.4f}", f"{col} <= {key}")
'''

graph = graphviz.Source(dot_data) 
graph.render('kk.img',format='png', view=False)