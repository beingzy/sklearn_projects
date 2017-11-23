"""build classifiers to classify XOR
"""
import os
import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC


def gen_xor_data(size, random_seed=None):
    """random xor sample
    """
    def assign_label(a, b):
        """return 1 if a != b, or 0 elsewhere
        """
        a_isgt = a > 0.5
        b_isgt = b > 0.5

        if a_isgt == b_isgt:
            return 1
        else:
            return 0

    if random_seed is not None:
        np.random.seed(random_seed)

    x1 = np.random.uniform(size=size)
    x2 = np.random.uniform(size=size)
    y  = [assign_label(ii, jj) for (ii, jj) in zip(x1, x2)]

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y
    })


def classifier_evaluation(ytrue, ypred):
    """function compute key performance metrics
    """
    from sklearn.metrics.classification import (
        accuracy_score,
        precision_score,
        recall_score
    )

    return {
        "accuracy_score": accuracy_score(ytrue, ypred),
        "precision_score": precision_score(ytrue, ypred),
        "recall_score": recall_score(ytrue, ypred)
    }


## =================
##
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
