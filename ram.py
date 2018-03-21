import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os
import glob

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio


from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D, Input
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras import optimizers, metrics


from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

# Todo:
# 1. Glimpse Sensor
#    (a) glipmse network
#    (b) location network
# 2. Internal State, the hidden state of RNN and get updated by core network -> h_t = f_h(h_t-1, g_t ; theta_h) 
#    (a) First input: g_t from glimpse network
#    (b) Second input: hidden state in previous recurrent unit
#    (c) output: actions
# 3. Actions
#    (a) Input: Internal State
#    (b) First output: l_t, location to deploy its sensor
#    (c) Second output: classification, conditioned on both hidden state then theta_a. find out what is theta_a
# 4. Reward
#    (a) reward signal, r_t + 1
#    (b) accumulate and attempt to maximize the r, each correct classification moves reward closer to 1.
#        - 



def create_glimpse(image, scale, no_patches, lt):

	lt_x = lt[0][0]
	lt_y = lt[0][1]
	bound_x = image.shape[0]
	bound_y = image.shape[1]
	glimpse_seq = np.empty([0])

	for i in (range(no_patches)):
		current_scale = scale * (i + 1)


		from_x, to_x = int(lt_x - current_scale), int(lt_x + current_scale)
		from_y, to_y = int(lt_y - current_scale), int(lt_y + current_scale)
		
		# check boundary
		if from_x < 0:
			from_x = 0
		if from_y < 0:
			from_y = 0
		if to_x > bound_x:
			to_x = bound_x
		if to_y > bound_y:
			to_y = bound_y
		# print("from_x: %i, to_x: %i, from_y: %i, to_y: %i" % (from_x, to_x, from_y, to_y))

		cropped_image = image[from_x:to_x, from_y:to_y]

		glimpse = cropped_image
		glimpse = cv2.resize(glimpse, (scale, scale))
		if i == 0:
			glimpse_seq = glimpse
		else:
			glimpse_seq = np.concatenate((glimpse_seq, glimpse), axis=2)
		# cv2.imshow("cropped", glimpse)
		# cv2.waitKey(0)	
	glimpse_seq = glimpse_seq.reshape((1, scale, scale, no_patches * 3))
	print(glimpse_seq.shape)
	return glimpse_seq

def Glimpse_Network(glimpse_seq, lt_previous, size):

	# glimpse_model = Sequential()
	# glimpse_model.add(Flatten(input_shape=(size, size, 3)))
	# glimpse_model.add(Dense(4096, activation = 'relu'))

	########## theta g^0, glimpse encoding ###########	
	adam = optimizers.Adam(lr=0.00001, decay=0.000001)
	glimpse_input = Input(shape=(size, size, 12))
	glimpse_flatten = Flatten()(glimpse_input)
	glimpse_fc = Dense(112, activation = 'relu')(glimpse_flatten)

	glimpse_model = Model(inputs = glimpse_input, outputs = glimpse_fc)
	glimpse_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])


	########## theta g^1, location encoding ###########
	location_input = Input(shape=(2, ))
	location_fc = Dense(112, activation = 'relu')(location_input)

	location_model = Model(inputs = location_input, outputs = location_fc)
	location_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])


	# glimpse_model.fit(glimpse_seq, glimpse_seq)
	item = glimpse_model.predict(glimpse_seq)
	loc = location_model.predict(lt_previous)
	# print(loc)
	return glimpse_model
	
# def location_network(ht):



# def recurrent_network(ht_prev, gt):




imagename = 'asd.png'
image = cv2.imread(imagename)
image = cv2.resize(image, (224,224))
scale = 56
no_patches = int(224/scale) 
lt = np.array(([120], [50]))
lt = lt.transpose()
print(lt.shape)
glimpse_seq = create_glimpse(image, scale, no_patches, lt)
Glimpse_Network(glimpse_seq, lt, scale)


# foveated_image = foveated_image.reshape(1, 224, 224, 3)
# Glimpse_Network(foveated_image, 224)
# adam = optimizers.Adam(lr=0.00001, decay=0.000001)

# initial_t = [1, 1]
# Glimpse_Network(foveated_image, initial_t, 224)
