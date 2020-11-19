#!/usr/bin/env python
# coding: utf-8

# # Classification Algorithms
# #### By Zachary Austin Ellis & Jose Carlos Gomez-Vazquez
# 
# In this notebook we will implement our own version of the Decision Tree Classifier, Random Forest Classifer, and Naive Bayes Classifier then compare their performance against SciKit-Learn's implementations.
# For these algorithms, the Red Wine Quality dataset provided by Kaggle will be used.
# 
# Source: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

# In[34]:


import numpy as np
import pandas as pd

# Used to compare our implementation's performance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ### Preprocessing

# In[35]:


def train_test_split(df, test_size):
    testing_set = df.sample(test_size)
    df.drop(index=testing_set.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    X_train = df.drop(columns="quality")
    X_test = testing_set.drop(columns="quality").reset_index(drop=True)
    
    Y_train = df["quality"]
    Y_test = testing_set["quality"].reset_index(drop=True)
    
    return X_train, X_test, Y_train, Y_test


wine_data = pd.read_csv("../data/winequality-red.csv")
print("\t\t\t\t\tRed Wine Data\n", wine_data)

X_train, X_test, Y_train, Y_test = train_test_split(wine_data.copy(), 100)


# ### Decision Tree Classifier

# In[36]:


class DecisionTree:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass


# ### Decision Tree Comparision

# In[37]:


DecisionTreeA = DecisionTree()
DecisionTreeB = DecisionTreeClassifier()


# ### Random Forest Classifier

# In[38]:


class RandomForest:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass


# ### Random Forest Comparison

# In[39]:


RandomForestA = DecisionTree()
RandomForestB = RandomForestClassifier()


# ### Naive Bayes Classifier

# In[40]:


class NaiveBayes:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass


# ### Naive Bayes Comparison

# In[41]:


NaiveBayesA = NaiveBayes()
NaiveBayesB = GaussianNB()

