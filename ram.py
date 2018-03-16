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



def image_foveate(image, scale, no_patches):

	image_centre = image.shape[0]
	for i in (range(no_patches)):
		current_scale = scale * (i + 1)
		from_x, to_x = int(image_centre - current_scale), int(image_centre + current_scale)
		from_y, to_y = int(image_centre - current_scale), int(image_centre + current_scale)
		cropped_image = image[from_x:to_x, from_y:to_y]
		print("from_x = %i, to_x = %i" % (from_x, to_x))
		print("from_y = %i, to_y = %i" % (from_y, to_y))

		foveated_image = image
		foveated_image = cv2.blur(foveated_image, (20, 20))
		foveated_image[from_x:to_x, from_y:to_y] = cropped_image
		# cv2.imshow("cropped", foveated_image)
		# cv2.waitKey(0)	

		return foveated_image

def Glimpse_Network(size):

	glimpse_model = Sequential()
	glimpse_model.add(Flatten(input_shape=(size, size, 3)))
	glimpse_model.add(Dense(4096))

	glimpse_model2 = Sequential()
	glimpse_model2.add(Flatten(input_shape=(size, size, 3)))
	glimpse_model2.add(Dense(4096))	

	return glimpse_model, glimpse_model2
	# glimpse_net = Model(inputs=foveated_image, outputs=glipmse_encoder)
	



imagename = 'asd.png'
image = cv2.imread(imagename)
image = cv2.resize(image, (224,224))
scale = 28
no_patches = int(224/scale/2) 
foveated_image = image_foveate(image, scale, no_patches)
foveated_image = foveated_image.reshape(1, 224, 224, 3)
# Glimpse_Network(foveated_image, 224)
adam = optimizers.Adam(lr=0.00001, decay=0.000001)
gn, gn2 = Glimpse_Network(size=224)
gn.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
# gn2.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
gn.fit(foveated_image, '1')
# glimpse_net = Model(inputs=gn.input, outputs=gn.output)
# glimpse_net.predict(foveated_image)

# glimpse_net = Model(inputs=gn2.input, outputs=gn2.output)
# glimpse_net.predict(foveated_image)



