"""util functions
"""
import numpy as np
import pandas as pd


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
