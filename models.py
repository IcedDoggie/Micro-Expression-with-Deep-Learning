import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

def Recognition_LSTM():
	data_dim=r*w # 2500
	print(data_dim)
	timesteps=10	

	model=Sequential()
	# model.add(TimeDistributed(Dense(data_dim), input_shape=(timesteps, data_dim)))
	model.add(LSTM(2500, return_sequences=True, input_shape=(timesteps, data_dim)))
	model.add(LSTM(500,return_sequences=False))
	##model.add(LSTM(500,return_sequences=True))
	##model.add(LSTM(50,return_sequences=False))
	model.add(Dense(50,activation='sigmoid'))
	model.add(Dense(5,activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=[metrics.categorical_accuracy])

	return model

def CNN():
	print("Spatial Features Extraction.")
	

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
