import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import pydot, graphviz

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.applications.vgg16 import VGG16 as keras_vgg16
import keras

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, loading_smic_labels
from models import VGG_16


############################## Loading Labels & Images ##############################
# /media/ice/OS/Datasets/SMIC_TIM10/SMIC_TIM10
root_db_path = "/media/ice/OS/Datasets/"
dB = "SMIC_TIM10"
inputDir = root_db_path + dB + "/" + dB + "/"  
workplace = root_db_path + dB + "/"


subject, filename, label, num_frames = loading_smic_labels(root_db_path, dB)
filename = filename.as_matrix()
label = label.as_matrix()

table = np.transpose( np.array( [filename, label] ) )	

# os.remove(workplace + "Classification/SMIC_label.txt")

################# Variables #############################
spatial_size = 224
r = w = spatial_size
subjects = 16
samples = 164
n_exp = 3

IgnoredSamples_index = np.empty([0])
VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
listOfIgnoredSamples = []

timesteps_TIM = 10
data_dim = r * w
pad_sequence = 10

#########################################################


############## Flags ####################
resizedFlag = 1
train_spatial_flag = 1
train_temporal_flag = 1
svm_flag = 0
finetuning_flag = 1
tensorboard_flag = 0
cam_visualizer_flag = 1
#########################################

############## Reading Images and Labels ################
# SubperdB = Read_SMIC_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size)
SubperdB = Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size)

labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
######################################################################################


########### Model #######################
sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.00001)

if train_spatial_flag == 0 and train_temporal_flag == 1:
	data_dim = spatial_size * spatial_size
else:
	data_dim = 4096
temporal_model = Sequential()
temporal_model.add(LSTM(2622, return_sequences=True, input_shape=(10, data_dim)))
temporal_model.add(LSTM(1000, return_sequences=False))
temporal_model.add(Dense(128, activation='relu'))
temporal_model.add(Dense(5, activation='sigmoid'))
temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
#########################################



################# Pretrained Model ###################
vgg_model = VGG_16('VGG_Face_Deep_16.h5')
# keras_vgg = keras_vgg16(weights='imagenet')

# vgg_model = VGG_16('imagenet')
vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.sparse_categorical_accuracy])
plot_model(vgg_model, to_file='model.png', show_shapes=True)

svm_classifier = SVC(kernel='linear', C=1)
######################################################

########### Training Process ############
# Todo:
# 1) LOSO (done)
# 2) call model (done)
# 3) saving model architecture
# 4) Saving Checkpoint
# 5) make prediction (done)
if tensorboard_flag == 1:
	tensorboard_path = "/home/ice/Documents/Micro-Expression/tensorboard/"

tot_mat = np.zeros((n_exp,n_exp))
spatial_weights_name = 'vgg_spatial_ID_under_dev_smic.h5'
temporal_weights_name = 'temporal_ID_under_dev_smic.h5'
for sub in range(subjects):
	vgg_model = VGG_16('VGG_Face_Deep_16.h5')
	vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.sparse_categorical_accuracy])

	############ for tensorboard ###############
	if tensorboard_flag == 1:
		cat_path = tensorboard_path + str(sub) + "/"
		os.mkdir(cat_path)
		tbCallBack = keras.callbacks.TensorBoard(log_dir=cat_path, write_graph=True)

		cat_path2 = tensorboard_path + str(sub) + "spat/"
		os.mkdir(cat_path2)
		tbCallBack2 = keras.callbacks.TensorBoard(log_dir=cat_path2, write_graph=True)
	#############################################

	image_label_mapping = np.empty([0])

	Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects)

	# Rearrange Training labels into a vector of images, breaking sequence
	Train_X_spatial = Train_X.reshape(Train_X.shape[0]*10, r, w, 1)
	Test_X_spatial = Test_X.reshape(Test_X.shape[0]* 10, r, w, 1)

	# Extend Y labels 10 fold, so that all images have labels
	Train_Y_spatial = np.repeat(Train_Y, 10, axis=0)
	Test_Y_spatial = np.repeat(Test_Y, 10, axis=0)
	

	# Duplicate channel of input image
	Train_X_spatial = duplicate_channel(Train_X_spatial)
	Test_X_spatial = duplicate_channel(Test_X_spatial)
	

	# print ("Train_X_shape: " + str(np.shape(Train_X_spatial)))
	# print ("Train_Y_shape: " + str(np.shape(Train_Y_spatial)))
	# print ("Test_X_shape: " + str(np.shape(Test_X_spatial)))	
	# print ("Test_Y_shape: " + str(np.shape(Test_Y_spatial)))	
	# print(Train_X_spatial)
	##################### Training & Testing #########################

		

	X = Train_X_spatial.reshape(Train_X_spatial.shape[0], 3, r, w)
	y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], 5)

	test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], 3, r, w)
	test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], 5)

	###### conv weights must be freezed for transfer learning ######
	if finetuning_flag == 1:
		for layer in vgg_model.layers[:33]:
			layer.trainable = False

	if train_spatial_flag == 1 and train_temporal_flag == 1:
		# trains encoder until fc, train temporal
		
		# Spatial Training
		if tensorboard_flag == 1:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True, callbacks=[tbCallBack2])
		else:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True)

		vgg_model.save_weights(spatial_weights_name)
		model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
		plot_model(model, to_file="spatial_module_FULL_TRAINING.png", show_shapes=True)	

		# Spatial Encoding
		output = model.predict(X, batch_size = 1)
		features = output.reshape(int(output.shape[0]/10), 10, output.shape[1])
		
		# Temporal Training
		if tensorboard_flag == 1:
			temporal_model.fit(features, Train_Y, batch_size=1, epochs=1, callbacks=[tbCallBack])
		else:
			temporal_model.fit(features, Train_Y, batch_size=1, epochs=1)	

		temporal_model.save_weights(temporal_weights_name)

		# Testing
		output = model.predict(test_X, batch_size = 1)
		features = output.reshape(int(output.shape[0]/10), 10, output.shape[1])
		predict = temporal_model.predict_classes(features, batch_size=1)


	elif train_spatial_flag == 1 and train_temporal_flag == 0 and cam_visualizer_flag == 0:
		# trains spatial module ONLY, no escape

		# Spatial Training
		if tensorboard_flag == 1:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True, callbacks=[tbCallBack2])
		else:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True)

		vgg_model.save_weights(spatial_weights_name)
		plot_model(vgg_model, to_file="spatial_module_ONLY.png", show_shapes=True)

		# Testing
		predict = vgg_model.predict(test_X, batch_size = 1)
		Test_Y_gt = np.repeat(Test_Y_gt, 10, axis=0)


	elif train_spatial_flag == 0 and train_temporal_flag == 1:
		# trains temporal module ONLY.

		# Temporal Training
		if tensorboard_flag == 1:
			temporal_model.fit(Train_X, Train_Y, batch_size=1, epochs=1, callbacks=[tbCallBack])
		else:
			temporal_model.fit(Train_X, Train_Y, batch_size=1, epochs=1)	

		temporal_model.save_weights(temporal_weights_name)

		# Testing
		predict = temporal_model.predict_classes(Test_X, batch_size = 1)

	elif svm_flag == 1 and finetuning_flag == 0:
		# no finetuning

		X = vgg_model.predict(X, batch_size=1)
		y_for_svm = np.argmax(y, axis=1)

		svm_classifier.fit(X, y_for_svm)

		test_X = vgg_model.predict(test_X, batch_size=1)
		predict = svm_classifier.predict(test_X)


		Test_Y_gt = np.repeat(Test_Y_gt, 10, axis=0)

	elif train_spatial_flag == 1 and train_temporal_flag == 0 and cam_visualizer_flag == 1:
		# trains spatial module & CAM ONLY

		# Spatial Training
		if tensorboard_flag == 1:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True, callbacks=[tbCallBack2])
		else:
			vgg_model.fit(X, y, batch_size=1, epochs=1, shuffle=True)

		vgg_model.save_weights(spatial_weights_name)
		plot_model(vgg_model, to_file="spatial_module_CAM_ONLY.png", show_shapes=True)

		# Testing
		predict = vgg_model.predict(test_X, batch_size = 1)	
		Test_Y_gt = np.repeat(Test_Y_gt, 10, axis=0)		

	##############################################################

	#################### Confusion Matrix Construction #############
	print (predict)
	print (Test_Y_gt)	

	ct = confusion_matrix(Test_Y_gt,predict)
	# check the order of the CT
	order = np.unique(np.concatenate((predict,Test_Y_gt)))
	
	# create an array to hold the CT for each CV
	mat = np.zeros((n_exp,n_exp))
	# put the order accordingly, in order to form the overall ConfusionMat
	for m in range(len(order)):
		for n in range(len(order)):
			mat[int(order[m]),int(order[n])]=ct[m,n]
		   
	tot_mat = mat + tot_mat
	################################################################
	
	#################### cumulative f1 plotting ######################
	microAcc = np.trace(tot_mat) / np.sum(tot_mat)
	[f1,precision,recall] = fpr(tot_mat,n_exp)


	file = open(workplace+'Classification/'+ 'Result/' + '/f1.txt', 'a')
	file.write(str(f1) + "\n")
	file.close()
	##################################################################

	################# write each CT of each CV into .txt file #####################
	record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects)
	###############################################################################