"""build classifiers to classify XOR
"""
import os
import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC

from utils import gen_xor_data
from utils import classifier_evaluation


## =================
## model building
## =================
sample_data = gen_xor_data(size=1000, random_seed=42)
xx = sample_data[['x1', 'x2']].as_matrix()
yy = sample_data['y'].as_matrix()

train_xx, test_xx, train_yy, test_yy = train_test_split(
    xx, yy, test_size=0.3)

clf = svm.SVC(kernel='rbf', probability=False)
clf.fit(train_xx, train_yy)

train_eval = classifier_evaluation(train_yy, clf.predict(train_xx))
test_eval = classifier_evaluation(test_yy, clf.predict(test_xx))
print("------SVM-------\n")
print('train performance: \n')
print(train_eval)
print('test performance: \n')
print(test_eval)


# LinearSVC
linear_clf = LinearSVC()
linear_clf.fit(train_xx, train_yy)

train_eval = classifier_evaluation(train_yy, linear_clf.predict(train_xx))
test_eval = classifier_evaluation(test_yy, linear_clf.predict(test_xx))
print("------LinaerSVC-------\n")
print('train performance: \n')
print(train_eval)
print('test performance: \n')
print(test_eval)
