#!/usr/bin/env python
# coding: utf-8

# # Classification Algorithms
# #### By Zachary Austin Ellis & Jose Carlos Gomez-Vazquez
# 
# In this notebook we will implement our own version of the Decision Tree Classifier, Random Forest Classifer, and Naive Bayes Classifier then compare their performance against SciKit-Learn's implementations.
# For these algorithms, the Red Wine Quality dataset provided by Kaggle will be used.
# 
# Source: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

# In[461]:


import math
import numpy as np
import pandas as pd

# Used to compare our implementation's performance
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ### Preprocessing

# In[462]:


def train_test_split(df, test_size=0):
    if test_size == 0:
        test_size = math.floor( len(df.index) * 0.3 )

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

X_train, X_test, Y_train, Y_test = train_test_split(wine_data.copy())


# ### Decision Tree Classifier

# In[463]:


def label_counts(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("parameter `data` must be a Pandas DataFrame")

    labels = data.quality.unique()
    counts = {}
    for label in labels:
        counts[str(label)] = len(data[data['quality'] == label].index)

    return counts

def gini(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("parameter `data` must be a Pandas DataFrame")

    counts = label_counts(data)
    impurity = 1.0
    for label in counts:
        probability = counts[label] / len(data.index)
        impurity -= probability**2

    return impurity

def info_gain(left_branch, right_branch, current_uncertainty):
    if not isinstance(left_branch, pd.DataFrame):
        raise TypeError("parameter `left_branch` must be a Pandas DataFrame")
    if not isinstance(right_branch, pd.DataFrame):
        raise TypeError("parameter `right_branch` must be a Pandas DataFrame")
    if not isinstance(current_uncertainty, float):
        raise TypeError("parameter `current_uncertainty` must be a floating point number")

    probability = len(left_branch.index) / (len(left_branch.index) + len(right_branch.index))
    return current_uncertainty - probability * gini(left_branch) - (1 - probability) * gini(right_branch)

def split(data, feature, split_point):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("parameter `data` must be a Pandas DataFrame")
    if not isinstance(feature, str):
        raise TypeError("parameter `feature` must be a String")
    if not isinstance(split_point, float):
        raise TypeError("parameter `split_point` must be a floating point number")

    true_branch = {}
    false_branch = {}
    for value in data.iterrows():
        if value[1][feature] >= split_point:
            true_branch[value[0]] = value[1]
        else:
            false_branch[value[0]] = value[1]

    return pd.DataFrame.from_dict(true_branch, orient='index').reset_index(drop=True), pd.DataFrame.from_dict(false_branch, orient='index').reset_index(drop=True)


class Node:
    """
    Represents a node on a decision tree
    """

    def __init__(self, is_leaf, **kwargs):
        if not isinstance(is_leaf, bool):
            raise TypeError("parameter `is_leaf` must be a Boolean")

        # Leaf Node requirements
        if is_leaf:
            if 'predictions' not in kwargs:
                raise ValueError("parameter `predictions` is required for leaf nodes")
            if not isinstance(kwargs['predictions'], pd.DataFrame):
                raise TypeError("parameter `predictions` must be a Pandas DataFrame")

        # Decision Node requirements
        else:
            if 'true_branch' not in kwargs:
                raise ValueError("parameter `true_branch` is required for non-leaf nodes")
            if 'false_branch' not in kwargs:
                raise ValueError("parameter `false_branch` is required for non-leaf nodes")
            if 'split_point' not in kwargs:
                raise ValueError("parameter `split_point` is required for non-leaf nodes")

            if not isinstance(kwargs['true_branch'], Node):
                raise TypeError("parameter `true_branch` must be a Node")
            if not isinstance(kwargs['false_branch'], Node):
                raise TypeError("parameter `false_branch` must be a Node")
            if not isinstance(kwargs['split_point'], tuple):
                raise TypeError("parameter `split_point` must be a tuple (feature, value)")
            if not isinstance(kwargs['split_point'][0], str):
                raise TypeError("parameter `split_point[0]` must be a String ([feature], value)")
            if not isinstance(kwargs['split_point'][1], float):
                raise TypeError("parameter `split_point[1]` must be a tuple (feature, [value])")

        self.is_leaf = is_leaf
        if is_leaf:
            self.predictions = kwargs['predictions']
            
            counts = label_counts(self.predictions)
            if len(counts) == 1:
                self.label = self.predictions['quality'][0]
            else:
                most = 0
                self.label = None
                for label in counts:
                    if counts[label] > most:
                        most = counts[label]
                        self.label = int(label)

        else:
            self.true_branch = kwargs['true_branch']
            self.false_branch = kwargs['false_branch']
            self.split_point = kwargs['split_point']

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf Node\n\n{self.predictions}\n"
        else:
            return f"Decision Node: Split at feature `{self.split_point[0]}` with value {self.split_point[1]}\n{self.true_branch}{self.false_branch}\n"


class DecisionTree:
    def __init__(self):
        self.tree = None
        self.original_data = None

    def fit(self, X, Y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("parameter `X` must be a Pandas DataFrame")
        if not isinstance(Y, pd.DataFrame) and not isinstance(Y, pd.Series):
            raise TypeError("parameter `Y` must be a Pandas DataFrame or Pandas Series")

        self.original_data = (X, Y)

        # Re-merge quality row into dataset (not-ideal, but necessary for this implementaion)
        data = X.copy()
        data.insert(len(data.columns), "quality", Y)
        self.tree = self.__build_tree(data)

        return

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("parameter `X` must be a Pandas DataFrame")

        return [self.__classify(data[1], self.tree) for data in X.iterrows()]

    def __build_tree(self, data):
        split_point, gain = self.__best_split(data)

        if gain == 0:
            return Node(True, predictions=data)

        true_data, false_data = split(data, split_point[0], split_point[1])
        true_branch = self.__build_tree(true_data)
        false_branch = self.__build_tree(false_data)

        return Node(False, true_branch=true_branch, false_branch=false_branch, split_point=split_point)

    def __best_split(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("parameter `data` must be a Pandas DataFrame")

        best_info_gain = 0.0
        best_split_point = None
        current_uncertainty = gini(data)

        for feature in data:
            # Dont split on quality (again, not-ideal)
            if feature == 'quality':
                continue

            values = data[feature].unique()
            for value in values:
                true_branch, false_branch = split(data, feature, value)

                # Skip split if no split occurred
                if len(true_branch.index) == 0 or len(false_branch.index) == 0:
                    continue

                gain = info_gain(true_branch, false_branch, current_uncertainty)
                if gain >= best_info_gain:
                    best_info_gain = gain
                    best_split_point = (feature, value)

        return best_split_point, best_info_gain

    def __classify(self, data, node):
        if not isinstance(data, pd.Series):
            raise TypeError("parameter `data` must be a Pandas Series")
        if not isinstance(node, Node):
            raise TypeError("parameter `node` must be a Node")

        if node.is_leaf:
            return node.label

        feature, value = node.split_point
        if data[feature] >= value:
            return self.__classify(data, node.true_branch)
        else:
            return self.__classify(data, node.false_branch)

    def __repr__(self):
        if self.tree != None:
            return f"{self.tree}"
        else:
            return "Tree has not been trained"


# ### Decision Tree Comparision

# In[ ]:


DecisionTreeA = DecisionTree()
DecisionTreeA.fit(X_train, Y_train)
print("Tree A\n\n", DecisionTreeA)
predictionsA = DecisionTreeA.predict(X_test)
print("Predictions A\n", predictionsA)

DecisionTreeB = DecisionTreeClassifier()
DecisionTreeB = DecisionTreeB.fit(X_train, Y_train)
predictionsB = DecisionTreeB.predict(X_test)
print("Predictions B\n", predictionsB)


# ### Random Forest Classifier

# In[ ]:


class RandomForest:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass


# ### Random Forest Comparison

# In[ ]:


RandomForestA = RandomForest()
RandomForestB = RandomForestClassifier()


# ### Naive Bayes Classifier and Comparison

# In[ ]:


from random import randrange
from math import sqrt
from math import exp
from math import pi

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
 
# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

NaiveBayesB = GaussianNB()
NaiveBayesB.fit(NB_X_train, NB_Y_train)
NB_Y_pred = NaiveBayesB.predict(NB_X_test)
print ("Scikit-learn GaussianNB Accuracy: {0:.3f}".format(accuracy_score(NB_Y_test, NB_Y_pred)))

dataset = pd.read_csv("../data/winequality-red-no-header.csv")
datalist = dataset.values.tolist()

n_folds = 5
scores = evaluate_algorithm(datalist, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Naive Bayes (from scratch) Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# ### Naive Bayes Comparison

# In[ ]:


#NaiveBayesA = NaiveBayes()
NaiveBayesB = GaussianNB()

model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print ("Scikit-learn GaussianNB accuracy: {0:.3f}".format(accuracy_score(Y_test, Y_pred)))

