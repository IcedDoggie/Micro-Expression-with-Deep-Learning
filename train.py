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
from PIL import Image

from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.applications.vgg16 import VGG16 as keras_vgg16
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import keras
from keras.callbacks import EarlyStopping

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import loading_smic_table, loading_casme_table, loading_samm_table, ignore_casme_samples, ignore_casmergb_samples # data loading scripts
from utilities import record_loss_accuracy, record_weights, record_scores, LossHistory # recording scripts
from list_databases import load_db, restructure_data
from models import VGG_16, temporal_module, VGG_16_4_channels, convolutional_autoencoder


def train(batch_size, spatial_epochs, temporal_epochs, train_id, dB, spatial_size, flag, tensorboard):
	############## Path Preparation ######################
	root_db_path = "/media/ice/OS/Datasets/"
	tensorboard_path = "/home/ice/Documents/Micro-Expression/tensorboard/"

	######################################################

	############## Variables ###################

	r, w, subjects, samples, n_exp, VidPerSubject, timesteps_TIM, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples, db_home, db_images = load_db(root_db_path, dB, spatial_size)

	# total confusion matrix to be used in the computation of f1 score
	tot_mat = np.zeros((n_exp,n_exp))

	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min')

	############################################

	############## Flags ####################
	tensorboard_flag = tensorboard
	resizedFlag = 1
	train_spatial_flag = 0
	train_temporal_flag = 0
	svm_flag = 0
	finetuning_flag = 0
	cam_visualizer_flag = 0
	channel_flag = 0

	if flag == 'st':
		train_spatial_flag = 1
		train_temporal_flag = 1
		finetuning_flag = 1
	elif flag == 's':
		train_spatial_flag = 1
		finetuning_flag = 1
	elif flag == 't':
		train_temporal_flag = 1
	elif flag == 'nofine':
		svm_flag = 1
	elif flag == 'scratch':
		train_spatial_flag = 1
		train_temporal_flag = 1
	elif flag == 'st4se':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 1
	elif flag == 'st7se':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 2
	elif flag == 'st4te':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 3
	elif flag == 'st7te':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 4

	#########################################

	############ Reading Images and Labels ################
	SubperdB = Read_Input_Images(db_images, listOfIgnoredSamples, dB, resizedFlag, table, db_home, spatial_size, channel)
	print("Loaded Images into the tray...")
	labelperSub = label_matching(db_home, dB, subjects, VidPerSubject)
	print("Loaded Labels into the tray...")

	if channel_flag == 1:
		SubperdB_strain = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table, db_home, spatial_size, 1)

	elif channel_flag == 2:	
		SubperdB_strain = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_TIM_Strain_TIM10', resizedFlag, table, db_home, spatial_size, 1)
		SubperdB_gray = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, db_home, spatial_size, 1)	

	elif channel_flag == 3:
		SubperdB_strain = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_TIM_Strain_TIM10', resizedFlag, table, db_home, spatial_size, 3)

	elif channel_flag == 4: 
		SubperdB_strain = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_TIM_Strain_TIM10', resizedFlag, table, db_home, spatial_size, 3)
		SubperdB_gray = Read_Input_Images(db_images, listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, db_home, spatial_size, 3)	
	#######################################################


	########### Model Configurations #######################
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=0.00001, decay=0.000001)

	# Different Conditions for Temporal Learning ONLY
	if train_spatial_flag == 0 and train_temporal_flag == 1 and dB != 'CASME2_Optical':
		data_dim = spatial_size * spatial_size
	elif train_spatial_flag == 0 and train_temporal_flag == 1 and dB == 'CASME2_Optical':
		data_dim = spatial_size * spatial_size * 3
	elif channel_flag == 3:
		data_dim = 8192
	elif channel_flag == 4:
		data_dim = 12288
	else:
		data_dim = 4096

	########################################################




	########### Training Process ############

	for sub in range(subjects):

		############### Reinitialization & weights reset of models ########################
		temporal_model = temporal_module(data_dim=data_dim, timesteps_TIM=timesteps_TIM)
		temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

		conv_ae = convolutional_autoencoder(spatial_size = spatial_size, classes = n_exp)
		conv_ae.compile(loss='binary_crossentropy', optimizer=adam)

		if channel_flag == 1 or channel_flag == 2:
			vgg_model = VGG_16_4_channels(classes=n_exp, spatial_size = spatial_size)
			vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		else:
			vgg_model = VGG_16(spatial_size = spatial_size, classes=n_exp, weights_path='VGG_Face_Deep_16.h5')
			vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

		svm_classifier = SVC(kernel='linear', C=1)
		####################################################################################
		
		
		############ for tensorboard ###############
		if tensorboard_flag == 1:
			cat_path = tensorboard_path + str(sub) + "/"
			os.mkdir(cat_path)
			tbCallBack = keras.callbacks.TensorBoard(log_dir=cat_path, write_graph=True)

			cat_path2 = tensorboard_path + str(sub) + "spat/"
			os.mkdir(cat_path2)
			tbCallBack2 = keras.callbacks.TensorBoard(log_dir=cat_path2, write_graph=True)
		#############################################

		Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects, n_exp)

		# Rearrange Training labels into a vector of images, breaking sequence
		Train_X_spatial = Train_X.reshape(Train_X.shape[0]*timesteps_TIM, r, w, channel)
		Test_X_spatial = Test_X.reshape(Test_X.shape[0]* timesteps_TIM, r, w, channel)

		# Extend Y labels 10 fold, so that all images have labels
		Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
		Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)	
		
		# Special Loading for 4-Channel
		if channel_flag == 1:
			Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects, n_exp)
			Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0]*timesteps_TIM, r, w, 1)
			Test_X_Strain = Test_X_Strain.reshape(Test_X.shape[0]*timesteps_TIM, r, w, 1)
		
			# Concatenate Train X & Train_X_Strain
			Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain), axis=3)
			Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain), axis=3)

			channel = 4

		elif channel_flag == 2:
			Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects, n_exp)
			Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0]*timesteps_TIM, r, w, 1)
			Test_X_Strain = Test_X_Strain.reshape(Test_X_Strain.shape[0]*timesteps_TIM, r, w, 1)

			Train_X_gray, _, Test_X_gray, _, _ = data_loader_with_LOSO(sub, SubperdB_gray, labelperSub, subjects)
			Train_X_gray = Train_X_gray.reshape(Train_X_gray.shape[0]*timesteps_TIM, r, w, 3)
			Test_X_gray = Test_X_gray.reshape(Test_X_gray.shape[0]*timesteps_TIM, r, w, 3)

			# Concatenate Train_X_Strain & Train_X & Train_X_gray
			Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain, Train_X_gray), axis=3)
			Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain, Test_X_gray), axis=3)	
			channel = 7		
		
		if channel == 1:
			# Duplicate channel of input image
			Train_X_spatial = duplicate_channel(Train_X_spatial)
			Test_X_spatial = duplicate_channel(Test_X_spatial)

	

		##################### Training & Testing #########################

		X = Train_X_spatial.reshape(Train_X_spatial.shape[0], channel, r, w)
		y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], n_exp)
		normalized_X = X.astype('float32') / 255.

		test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], channel, r, w)
		test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], n_exp)
		normalized_test_X = test_X.astype('float32') / 255.

		print(X.shape)

		###### conv weights must be freezed for transfer learning ######
		if finetuning_flag == 1:
			for layer in vgg_model.layers[:33]:
				layer.trainable = False

		if train_spatial_flag == 1 and train_temporal_flag == 1:

			# Spatial Training
			if tensorboard_flag == 1:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
			else:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[history, stopping])

			
			# record f1 and loss
			record_loss_accuracy(db_images, train_id, dB, history, 's')		

			# save vgg weights
			model = record_weights(vgg_model, spatial_weights_name, sub)

			# Spatial Encoding
			output = model.predict(X, batch_size = batch_size)
			features = output.reshape(int(Train_X.shape[0]), timesteps_TIM, output.shape[1])
			
			# Temporal Training
			if tensorboard_flag == 1:
				temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs, callbacks=[tbCallBack])
			else:
				temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs)	

			# save temporal weights
			temporal_model = record_weights(temporal_model, temporal_wei, subject, 't')

			# Testing
			output = model.predict(test_X, batch_size = batch_size)
			features = output.reshape(Test_X.shape[0], timesteps_TIM, output.shape[1])

			predict = temporal_model.predict_classes(features, batch_size=batch_size)

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

		file = open(db_home+'Classification/'+ 'Result/'+dB+'/f1_' + str(train_id) +  '.txt', 'a')
		file.write(str(f1) + "\n")
		file.close()
		##################################################################

		################# write each CT of each CV into .txt file #####################
		record_scores(db_home, dB, ct, sub, order, tot_mat, n_exp, subjects)
		###############################################################################
		