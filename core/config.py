# -*- coding: utf-8 -*-

'''This module allows to configure data processing behavior'''
#-------------------------------------------------------------------------------
# Classifiers
#-------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier

#-------------------------------------------------------------------------------
# Scalers
#-------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from core.scaler import MyScaler, IdentityScaler


DATA_SOURCE  = "./data/student_data.csv"

dict_classifier_grid = {
    "LogisticRegression": {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 20, 100],
        'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear', 'saga'],
        'max_iter': [15000]
    }
    , "ComplementNB": {
        'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'fit_prior': [True, False],
    }
    , "SVC": {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'probability': [True]
    }
    , "MultinomialNB": {
        'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'fit_prior': [True, False],
    }
    , "BernoulliNB": {
        'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'fit_prior': [True, False],
    }
    , "DecisionTreeClassifier": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }
}
CLASSIFIER = DecisionTreeClassifier
SCALER = IdentityScaler
SCALER_IMPROVE = MinMaxScaler
DATAFRAME_SELECTION = 'all'
CLASS_LABEL = 1
HAS_CALIBRATION = True
#-------------------------------------------------------------------------------
