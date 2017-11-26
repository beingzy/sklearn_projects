"""build one-layer neural network to approximate
   XOR
"""
import numpy as np
import pandas as pd

import sklearn as sk
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from utils import gen_xor_data
from utils import classifier_evaluation


# processing data
sample_data = gen_xor_data(size=1000, random_seed=42)
xx = sample_data[['x1', 'x2']].as_matrix()
yy = sample_data['y'].as_matrix()
# split datasets
train_xx, test_xx, train_yy, test_yy = train_test_split(
    xx, yy, test_size=0.3)

# build neural network
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(train_xx,
          train_yy,
          epochs=20,
          batch_size=50,
          verbose=1)

train_y_pred = model.predict(train_xx).flatten()
test_y_pred = model.predict(test_xx).flatten()
# train_eval = classifier_evaluation(train_yy, train_y_pred)
# test_eval = classifier_evaluation(test_yy, test_y_pred)
#print("------SVM-------\n")
#print('train performance: \n')
#print(train_eval)
#print('test performance: \n')
#print(test_eval)
