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
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D, Input, Concatenate, SimpleRNN
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras import optimizers, metrics
from keras.initializers import Zeros, RandomUniform
	
from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import LossHistory
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

history = LossHistory()



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
	# print(glimpse_seq.shape)
	return glimpse_seq



def glimpse_network(glimpse_seq, lt_prev, size, optimize):

	########## theta g^0, glimpse encoding ###########	
	glimpse_input = Input(shape=(size, size, 12))
	glimpse_flatten = Flatten()(glimpse_input)
	glimpse_fc = Dense(112, activation = 'relu')(glimpse_flatten)

	glimpse_model = Model(inputs = glimpse_input, outputs = glimpse_fc)
	glimpse_model.compile(loss='categorical_crossentropy', optimizer = optimize, metrics=[metrics.categorical_accuracy])


	########## theta g^1, location encoding ###########
	location_input = Input(shape=(2, ))
	location_fc = Dense(112, activation = 'relu')(location_input)

	location_model = Model(inputs = location_input, outputs = location_fc)
	location_model.compile(loss='categorical_crossentropy', optimizer = optimize, metrics=[metrics.categorical_accuracy])

	########## theta g^2, gt vector encoding ###########
	gt_concat = Concatenate(axis = 1)([glimpse_fc, location_fc])
	gt_fc = Dense(224, activation = 'relu')(gt_concat)

	gt_model = Model(inputs = [glimpse_input, location_input], outputs = gt_fc)
	gt_model.compile(loss='categorical_crossentropy', optimizer = optimize, metrics=[metrics.categorical_accuracy])

	gt = gt_model.predict([glimpse_seq, lt_prev])
	# print(gt.shape)

	return gt

def recurrent_network(gt, optimize):
	rnn_input = Input(shape=(1, 224))
	out, ht = SimpleRNN(256, return_state = True, activation = 'relu')(rnn_input)


	rnn_model = Model(inputs = rnn_input, outputs = [out, ht])
	rnn_model.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=[metrics.categorical_accuracy])

	# ht_prev = np.random.normal(size = (1, 1, 224))
	gt = gt.reshape(1, 1, 224)
	output, ht = rnn_model.predict(gt)
	# print(ht)
	# print(ht.shape)

	return output, ht



def location_network(ht, optimize):
	location_input = Input(shape=(256,))
	location_output = Dense(2, activation = 'tanh')(location_input)

	location_model = Model(inputs = location_input, outputs = location_output)
	location_model.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=[metrics.categorical_accuracy])

	mu = location_model.predict(ht)
	noise = np.random.normal(scale = 0.17, size = mu.shape)
	# print(mu)
	# print(noise)
	l_t = mu + noise
	# print(l_t)
	# print(l_t.shape)

	l_t = K.clip(l_t, min_value = -1, max_value = 1)
	l_t = K.stop_gradient(l_t)
	l_t = l_t.eval()
	# print(l_t)
	# print(l_t.shape)

	# normal sampling
	log_pi = np.random.normal(mu, 0.17)
	log_pi = sum(log_pi)
	log_pi = log_pi.reshape(log_pi.shape[0], 1)

	print(noise)
	print(log_pi)
	return l_t, log_pi

def action_network(ht, y, optimize):
	action_input = Input(shape=(256,))
	action_out = Dense(5, activation = 'softmax')(action_input)

	action_model = Model(inputs = action_input, outputs = action_out)
	action_model.compile(loss='logcosh', optimizer=optimize, metrics=[metrics.categorical_accuracy])

	action_model.fit(ht, y, callbacks=[history])
	action_loss = history.losses
	predicted_class = action_model.predict(ht)
	# print(predicted_class)
	# print(predicted_class.shape)

	return predicted_class, action_loss

def baseline_network(h_t, reward, optimize):
	baseline_input = Input(shape=(256, ))
	baseline_fc = Dense(5, activation = 'relu')(baseline_input)

	baseline_model = Model(inputs = baseline_input, outputs = baseline_fc)
	baseline_model.compile(loss='mean_squared_error', optimizer = optimize, metrics = [metrics.categorical_accuracy])

	baseline_model.fit(h_t, reward, callbacks=[history])
	baseline_loss = history.losses
	b_t = baseline_model.predict(h_t)

	return b_t, baseline_loss

# def reinforcement_objective():
	




imagename = 'asd.png'
image = cv2.imread(imagename)
image = cv2.resize(image, (224,224))
label = np.array([[0, 1, 2, 3, 4]])
scale = 56
no_patches = int(224/scale) 
l_t = np.array(([120], [50]))
l_t = l_t.transpose()
print(l_t.shape)

adam = optimizers.Adam(lr=0.00001, decay=0.000001)

glimpse_seq = create_glimpse(image, scale, no_patches, l_t)
g_t = glimpse_network(glimpse_seq, l_t, 56, adam)
output, h_t = recurrent_network(g_t, adam)
p_class, loss_action = action_network(h_t, label, adam)
l_t, log_pi = location_network(h_t, adam)


p_class = max(p_class)
reward = K.cast((p_class == label), dtype = 'float32')
reward = reward.eval()

b_t, baseline_loss = baseline_network(h_t, reward, adam)

# reinforce loss
adjusted_reward = reward - b_t
reinforce_loss = np.mean(-log_pi * adjusted_reward)

# hybrid loss
hybrid_loss = loss_action[0] + baseline_loss[0] + reinforce_loss



print(reward)
print(loss_action)
print(baseline_loss)
print(adjusted_reward)
print(reinforce_loss)
print(hybrid_loss)