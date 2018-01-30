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
from utilities import record_scores, loading_smic_table, loading_casme_table, ignore_casme_samples, ignore_casmergb_samples, LossHistory
from utilities import loading_samm_table
from models import VGG_16, temporal_module, modify_cam, VGG_16_4_channels, convolutional_autoencoder


def train_cae_lstm(batch_size, spatial_epochs, temporal_epochs, train_id, dB, spatial_size, flag, tensorboard):
	############## Path Preparation ######################
	root_db_path = "/media/ice/OS/Datasets/"
	workplace = root_db_path + dB + "/"
	inputDir = root_db_path + dB + "/" + dB + "/" 
	######################################################
	classes = 5
	if dB == 'CASME2_TIM':
		table = loading_casme_table(workplace + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(inputDir)

		############## Variables ###################
		r = w = spatial_size
		subjects=2
		samples = 246
		n_exp = 5
		# VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		listOfIgnoredSamples = []
		VidPerSubject = [2,1]
		timesteps_TIM = 10
		data_dim = r * w
		pad_sequence = 10
		channel = 3
		############################################		

		os.remove(workplace + "Classification/CASME2_TIM_label.txt")



	elif dB == 'CASME2_Optical':
		table = loading_casme_table(workplace + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index, _ = ignore_casme_samples(inputDir)

		############## Variables ###################
		r = w = spatial_size
		subjects=2
		samples = 246
		n_exp = 5
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 9
		data_dim = r * w
		pad_sequence = 9
		channel = 3
		############################################		

		# os.remove(workplace + "Classification/CASME2_TIM_label.txt")

	elif dB == 'CASME2_RGB':
		# print(inputDir)
		table = loading_casme_table(workplace + 'CASME2_RGB/CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casmergb_samples(inputDir)
		############## Variables ###################
		r = w = spatial_size
		subjects=26
		samples = 245 # not used, delete it later
		n_exp = 5
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w 
		pad_sequence = 10
		channel = 3
		############################################

	elif dB == 'SMIC_TIM10':
		table = loading_smic_table(root_db_path, dB)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 16
		samples = 164
		n_exp = 3
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		pad_sequence = 10
		channel = 1
		classes = 3
		#########################################################

	elif dB == 'SAMM_Optical':
		table, table_objective = loading_samm_table(root_db_path, dB)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 9
		data_dim = r * w
		pad_sequence = 10
		channel = 3
		classes = 8
		#########################################################	

	elif dB == 'SAMM_TIM10':
		table, table_objective = loading_samm_table(root_db_path, dB)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		pad_sequence = 10
		channel = 3
		classes = 8
		#########################################################			


	# print(VidPerSubject)

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
	elif flag == 'st4':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 1
	elif flag == 'st7':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 2
	#########################################



	############ Reading Images and Labels ################
	SubperdB = Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size, channel)
	print("Loaded Images into the tray...")
	labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
	print("Loaded Labels into the tray...")

	if channel_flag == 1:
		SubperdB_strain = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table, workplace, spatial_size, 1)
	elif channel_flag == 2:
		SubperdB_strain = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table, workplace, spatial_size, 1)
		SubperdB_gray = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, workplace, spatial_size, 3)		
	#######################################################


	########### Model Configurations #######################
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=0.00001, decay=0.000001)

	# Different Conditions for Temporal Learning ONLY
	if train_spatial_flag == 0 and train_temporal_flag == 1 and dB != 'CASME2_Optical':
		data_dim = spatial_size * spatial_size
	elif train_spatial_flag == 0 and train_temporal_flag == 1 and dB == 'CASME2_Optical':
		data_dim = spatial_size * spatial_size * 3
	else:
		data_dim = 4096

	########################################################


	########### Image Data Generator ##############
	image_generator = ImageDataGenerator(
		zca_whitening = True,
		rotation_range = 0.2,
		width_shift_range = 0.2,
		height_shift_range = 0.2, 
		zoom_range = 0.2,
		horizontal_flip = True,
		rescale = 1.5)
	###############################################

	########### Training Process ############
	# Todo:
	# 1) LOSO (done)
	# 2) call model (done)
	# 3) saving model architecture 
	# 4) Saving Checkpoint (done)
	# 5) make prediction (done)
	if tensorboard_flag == 1:
		tensorboard_path = "/home/ice/Documents/Micro-Expression/tensorboard/"

	# total confusion matrix to be used in the computation of f1 score
	tot_mat = np.zeros((n_exp,n_exp))

	# model checkpoint
	spatial_weights_name = 'vgg_spatial_'+ str(train_id) + '_' + str(dB) + '_'
	temporal_weights_name = 'temporal_ID_' + str(train_id) + '_' + str(dB) + '_'
	ae_weights_name = 'autoencoder_' + str(train_id) + '_' + str(dB) + '_'
	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min')



	for sub in range(subjects):
		############### Reinitialization & weights reset of models ########################

		vgg_model_cam = VGG_16(spatial_size=spatial_size, classes=classes, weights_path='VGG_Face_Deep_16.h5')

		temporal_model = temporal_module(data_dim=data_dim, classes=classes, timesteps_TIM=timesteps_TIM)
		temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

		conv_ae = convolutional_autoencoder(spatial_size = spatial_size, classes = classes)
		conv_ae.compile(loss='binary_crossentropy', optimizer=adam)

		if channel_flag == 1 or channel_flag == 2:
			vgg_model = VGG_16_4_channels(classes=classes, spatial_size = spatial_size)
			vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		else:
			vgg_model = VGG_16(spatial_size = spatial_size, classes=classes, weights_path='VGG_Face_Deep_16.h5')
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

		image_label_mapping = np.empty([0])


		Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects, classes)

		# Rearrange Training labels into a vector of images, breaking sequence
		Train_X_spatial = Train_X.reshape(Train_X.shape[0]*timesteps_TIM, r, w, channel)
		Test_X_spatial = Test_X.reshape(Test_X.shape[0]* timesteps_TIM, r, w, channel)

		
		# Special Loading for 4-Channel
		if channel_flag == 1:
			Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects, classes)
			Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0]*timesteps_TIM, r, w, 1)
			Test_X_Strain = Test_X_Strain.reshape(Test_X.shape[0]*timesteps_TIM, r, w, 1)
		
			# Concatenate Train X & Train_X_Strain
			Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain), axis=3)
			Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain), axis=3)

			channel = 4

		elif channel_flag == 2:
			Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects, classes)
			Train_X_gray, _, Test_X_gray, _, _ = data_loader_with_LOSO(sub, SubperdB_gray, labelperSub, subjects)
			Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0]*timesteps_TIM, r, w, 1)
			Test_X_Strain = Test_X_Strain.reshape(Test_X_Strain.shape[0]*timesteps_TIM, r, w, 1)
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

		# Extend Y labels 10 fold, so that all images have labels
		Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
		Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)		


		# print ("Train_X_shape: " + str(np.shape(Train_X_spatial)))
		# print ("Train_Y_shape: " + str(np.shape(Train_Y_spatial)))
		# print ("Test_X_shape: " + str(np.shape(Test_X_spatial)))	
		# print ("Test_Y_shape: " + str(np.shape(Test_Y_spatial)))	
		# print(Train_X_spatial)
		##################### Training & Testing #########################

		# print(Train_X_spatial.shape)	

		X = Train_X_spatial.reshape(Train_X_spatial.shape[0], channel, r, w)
		y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], classes)
		normalized_X = X.astype('float32') / 255.

		test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], channel, r, w)
		test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], classes)
		normalized_test_X = test_X.astype('float32') / 255.

		print(X.shape)

		###### conv weights must be freezed for transfer learning ######
		if finetuning_flag == 1:
			for layer in vgg_model.layers[:33]:
				layer.trainable = False
			for layer in vgg_model_cam.layers[:31]:
				layer.trainable = False

		if train_spatial_flag == 1 and train_temporal_flag == 1:
			# Autoencoder first training
			conv_ae.fit(normalized_X, normalized_X, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)		

			# remove decoder
			conv_ae.pop()
			conv_ae.pop()
			conv_ae.pop()
			conv_ae.pop()
			conv_ae.pop()
			conv_ae.pop()
			conv_ae.pop()


			# append dense layers
			conv_ae.add(Flatten())
			conv_ae.add(Dense(4096, activation='relu'))
			conv_ae.add(Dense(4096, activation='relu'))
			conv_ae.add(Dense(n_exp, activation='sigmoid'))
			model_ae = Model(inputs=conv_ae.input, outputs=conv_ae.layers[9].output)			
			plot_model(model_ae, to_file='autoencoders.png', show_shapes=True)

			# freeze encoder
			for layer in conv_ae.layers[:6]:
				layer.trainable = False

			# finetune dense layers
			conv_ae.compile(loss='categorical_crossentropy', optimizer=adam)
			conv_ae.fit(normalized_X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)		

	
			model_ae = Model(inputs=conv_ae.input, outputs=conv_ae.layers[8].output)
			plot_model(model_ae, to_file='autoencoders.png', show_shapes=True)

			# Autoencoding
			output = model_ae.predict(normalized_X, batch_size = batch_size)

			# print(output.shape)
			features = output.reshape(int(Train_X.shape[0]), timesteps_TIM, output.shape[1])
			

			temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs)	

			temporal_model.save_weights(temporal_weights_name + str(sub) + ".h5")

			# Testing
			output = model_ae.predict(test_X, batch_size = batch_size)


			features = output.reshape(Test_X.shape[0], timesteps_TIM, output.shape[1])

			predict = temporal_model.predict_classes(features, batch_size=batch_size)




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


		file = open(workplace+'Classification/'+ 'Result/'+dB+'/f1_' + str(train_id) +  '.txt', 'a')
		file.write(str(f1) + "\n")
		file.close()
		##################################################################

		################# write each CT of each CV into .txt file #####################
		record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects)
		###############################################################################