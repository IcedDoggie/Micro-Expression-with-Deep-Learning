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

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM
from keras.optimizers import SGD

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

def LSTM_KAIST(weights_path=None):
	temporal_model = Sequential()
	temporal_model.add(LSTM(512, return_sequences=True, input_shape=(5, 10)))
	temporal_model.add(LSTM(512, return_sequences=False))

	temporal_model.add(Dense(128, activation='sigmoid'))
	temporal_model.add(Dense(5, activation='sigmoid'))

	if weights_path:
		temporal_model.load_weights(weights_path)

	return temporal_model

def CNN_KAIST(weights_path=None):
	model = Sequential()
	model.add(Conv2D( 32, kernel_size=(1, 1), strides=(1,1), input_shape=(50, 50, 1) ))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	
	model.add(Conv2D( 64, kernel_size=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	
	model.add(Conv2D( 64, kernel_size=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	
	model.add(Dense( 512, activation='relu'))
	model.add(Dense( 512, activation='relu'))
	model.add(Flatten())
	model.add(Dense( 5, activation='softmax'))
	
	if weights_path:
		model.load_weights(weights_path)

	return model	


def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2622, activation='softmax'))


	if weights_path:
		model.load_weights(weights_path)

	model.add(Dense(5, activation='softmax'))
	return model

# VGG_16()