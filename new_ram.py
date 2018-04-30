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
import pydot, graphviz
from keras.utils import np_utils, plot_model


from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
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



def Recurrent_Attention_Network(input_img, input_loc, rnn_dim, timesteps, glimpse_fc_dim, h_fc_dim, input_dim):

	adam = optimizers.Adam(lr=0.00001, decay=0.000001)

	########## theta g^0, glimpse encoding ###########	
	glimpse_input = Input(shape=(input_dim, input_dim, 3))
	glimpse_flatten = Flatten()(glimpse_input)
	glimpse_fc = Dense(glimpse_fc_dim, activation = 'relu')(glimpse_flatten)

	########## theta g^1, location encoding ###########	
	location_input = Input(shape=(2, ))
	location_fc = Dense(glimpse_fc_dim, activation = 'relu')(location_input)

	########## theta g^2, gt vector encoding ###########
	gt_concat = Concatenate(axis = 1)([glimpse_fc, location_fc])
	gt_fc = Dense(glimpse_fc_dim * 2, activation = 'relu')(gt_concat)

	####### RNN ########
	rnn_input = Reshape((timesteps, glimpse_fc_dim * 2))(gt_fc)
	out, ht = SimpleRNN(rnn_dim, return_state = True, return_sequences = True, activation = 'relu')(rnn_input)

	############ classification ##############
	action_fc = Dense(h_fc_dim, activation = 'relu')(ht)
	action_softmax = Dense(5, activation = 'softmax')(action_fc)




	model = Model(inputs = [glimpse_input, location_input], outputs = [action_softmax])
	model.compile(loss=['categorical_crossentropy'], optimizer=adam, metrics = [metrics.categorical_accuracy])
	plot_model(model, to_file = "RAM.png", show_shapes=True)	



	return model


def location_network(ht):
	########### location ################
	adam = optimizers.Adam(lr=0.00001, decay=0.000001)
	ht_input = Input(shape=(1, 256))
	ht_flatten = Flatten()(ht_input)	
	location_output = Dense(2, activation = 'tanh')(ht_flatten)
	ht_model = Model(inputs = ht_input, outputs = location_output)
	ht_model.compile(loss = ['mean_squared_error'], optimizer = adam, metrics = [metrics.categorical_crossentropy])
	mu = ht_model.predict(ht)
	

	noise = np.random.normal(scale = 0.17, size = mu.shape)
	lt = mu + noise
	lt = K.clip(lt, min_value = -1, max_value = 1)
	lt = K.stop_gradient(lt)
	lt = lt.eval()

	log_pi = np.random.normal(mu, 0.17)
	log_pi = sum(log_pi)
	log_pi = log_pi.reshape(log_pi.shape[0], 1)

	# print(noise.shape)
	print(lt.shape)
	print(log_pi.shape)


	return lt, log_pi

baseline_history = LossHistory()
action_history = LossHistory()

image = cv2.imread('1.png')
image = cv2.resize(image, (224, 224))

image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

loc = np.array((112, 112))
loc = loc.reshape(1, 2)
y = np.array(([0, 0, 0, 0, 1]))
y = y.reshape(1, 5)

model = Recurrent_Attention_Network(image, loc, 256, 1, 112, 256, 224)	
model.fit([image, loc], [y], callbacks = [action_history])
ht_model = Model(inputs = model.input, outputs = model.layers[8].output)
out, ht = ht_model.predict([image, loc])
# print(ht.shape)

lt, log_pi = location_network(out)
print(lt)
print(log_pi)


# calculate reward
prediction = model.predict([image, loc])
prediction = max(prediction)
reward = K.cast((prediction == y), dtype = 'float32')
reward = reward.eval()

print(prediction)
print(reward)

# baseline
adam = optimizers.Adam(lr=0.00001, decay=0.000001)
model_baseline = Recurrent_Attention_Network(image, loc, 256, 1, 112, 256, 224)
model_baseline = Model(inputs = model_baseline.input, outputs = model_baseline.output)	
model_baseline.compile(loss=['mean_squared_error'], optimizer=adam, metrics = [metrics.categorical_accuracy])
model_baseline.fit([image, loc], [reward], callbacks = [baseline_history])
baseline = model_baseline.predict([image, loc])

# compute reinforce loss
adjusted_reward = reward - baseline
reinforce_loss = np.mean(-log_pi * adjusted_reward)

# hybrid loss
action_loss = action_history.losses[0]
baseline_loss = baseline_history.losses[0]
hybrid_loss = action_loss + baseline_loss + reinforce_loss
print(hybrid_loss)