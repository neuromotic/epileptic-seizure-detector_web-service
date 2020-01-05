#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:49:35 2019

@author: edwardmolina10
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from Plt_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import time
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from joblib import load

start_time = time.time()
dataset = pd.read_csv("/media/edwardmolina10/Personal1/Documentos/Maestría Ingeniería Telemática/Tesis/Epilepsy-Data-Analysis/Data/Features_Data/dataset_e3_features_epilepsy_resampled.csv", float_precision='high')
random_s = 3

X_dataset = dataset.loc[:,'acc_std_x' : 'bvp_fft_energy']
y_dataset = dataset['class_1']

# Split 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.25, random_state=random_s)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
dump(scaler, 'scaling.joblib')

clf = RandomForestClassifier().fit(X_train, y_train)
dump(clf, 'RandomForestClassifier.joblib')
print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

scaler_production = load('scaling.joblib')
X_test = scaler_production.transform(X_test)

classifier_production = load('RandomForestClassifier.joblib')
y_pred = classifier_production.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

class_names = np.array(['1', '0']).reshape(-1, 1)

np.set_printoptions(precision=2)
y_test = np.array(y_test)
y_pred = np.array(y_pred)

print("\nConfusion Matrix:\n")

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, normalize=True, title='Normalized confusion matrix')
plt.show()

print("Evaluation metrics:\n")
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
print('F1: {:.2f}'.format(f1_score(y_test, y_pred)))
print('Matthews correlation coefficient: {:.2f}'.format(matthews_corrcoef(y_test, y_pred)))