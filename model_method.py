import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report

# Import libarary confusion matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.model_selection import train_test_split

# Import libarary Logistic Regression
from sklearn.linear_model import LogisticRegression

from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

# Import libarary KNN
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

# Import libarary Support Vector Machines dan linier Support Vector Machines
from sklearn import svm
from sklearn.svm import LinearSVC

# Import libarary Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB


def predict(arr):
    # Load the model
    with open('final_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'Berat',1:'Ringan',2:'Sedang'}
    # return prediction as well as class probabilities
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)