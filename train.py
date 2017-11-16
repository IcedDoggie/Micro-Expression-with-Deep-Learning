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
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.applications.vgg16 import VGG16 as keras_vgg16
from keras.preprocessing.image import ImageDataGenerator
import keras

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, loading_smic_table, loading_casme_table, ignore_casme_samples, ignore_casmergb_samples
from models import VGG_16, temporal_module, modify_cam


def train(batch_size, spatial_epochs, temporal_epochs, train_id, dB, spatial_size, flag, tensorboard):
	############## Path Preparation ######################
	root_db_path = "/media/ice/OS/Datasets/"
	workplace = root_db_path + dB + "/"
	inputDir = root_db_path + dB + "/" + dB + "/" 
	######################################################

	if dB == 'CASME2_TIM':
		table = loading_casme_table(root_db_path + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(inputDir)

		############## Variables ###################
		r = w = spatial_size
		subjects=26
		samples = 246
		n_exp = 5
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		pad_sequence = 10
		############################################		

		os.remove(workplace + "Classification/CASME2_TIM_label.txt")

	elif dB == 'CASME2_Optical':
		table = loading_casme_table(root_db_path + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(inputDir)

		############## Variables ###################
		r = w = spatial_size
		subjects=26
		samples = 246
		n_exp = 5
		VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
		timesteps_TIM = 9
		data_dim = r * w
		pad_sequence = 9
		############################################		

		os.remove(workplace + "Classification/CASME2_TIM_label.txt")

	elif dB == 'CASME2_RGB':
		# print(inputDir)
		table = loading_casme_table('/media/viprlab/01D31FFEF66D5170/CASME2_RGB/CASME2_label_Ver_2.xls')
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
		#########################################################


	############## Flags ####################
	tensorboard_flag = tensorboard
	resizedFlag = 1
	train_spatial_flag = 0
	train_temporal_flag = 0
	svm_flag = 0
	finetuning_flag = 0
	cam_visualizer_flag = 0

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
	#########################################



	############ Reading Images and Labels ################
	SubperdB = Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size)
	print("Loaded Images into the tray...")
	labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
	print("Loaded Labels into the tray...")
	#######################################################


	########### Model Configurations #######################
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=0.00001)

	# Different Conditions for Temporal Learning ONLY
	if train_spatial_flag == 0 and train_temporal_flag == 1 and dB != 'CASME2_Optical':
		data_dim = spatial_size * spatial_size
	elif train_spatial_flag == 0 and train_temporal_flag == 1 and dB == 'CASME2_Optical':
		data_dim = spatial_size * spatial_size * 3
	else:
		data_dim = 4096

	########################################################


	########### Image Data Generator ##############
	# image_generator = ImageDataGenerator(
	# 	zca_whitening = True,
	# 	rotation_range = 0.2,
	# 	width_shift_range = 0.2,
	# 	height_shift_range = 0.2, 
	# 	zoom_range = 0.2,
	# 	horizontal_flip = True,
	# 	rescale = 1.5)
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
	spatial_weights_name = 'vgg_spatial_'+ str(train_id) + '_casme2_'
	temporal_weights_name = 'temporal_ID_' + str(train_id) + '_casme2_'


	for sub in range(subjects):
		############### Reinitialization & weights reset of models ########################
		vgg_model = VGG_16('VGG_Face_Deep_16.h5', spatial_size)
		vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.sparse_categorical_accuracy])

		vgg_model_cam = VGG_16('VGG_Face_Deep_16.h5', spatial_size)

		temporal_model = temporal_module(data_dim=data_dim, timesteps_TIM=timesteps_TIM)
		temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		
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

		Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects)

		# Rearrange Training labels into a vector of images, breaking sequence
		if dB == "CASME2_TIM" or dB == 'SMIC_TIM10':
			Train_X_spatial = Train_X.reshape(Train_X.shape[0]*10, r, w, 1)
			Test_X_spatial = Test_X.reshape(Test_X.shape[0]* 10, r, w, 1)
			# Duplicate channel of input image
			Train_X_spatial = duplicate_channel(Train_X_spatial)
			Test_X_spatial = duplicate_channel(Test_X_spatial)
			# Extend Y labels 10 fold, so that all images have labels
			Train_Y_spatial = np.repeat(Train_Y, 10, axis=0)
			Test_Y_spatial = np.repeat(Test_Y, 10, axis=0)			

		elif dB == 'CASME2_Optical':
			Train_X_spatial = Train_X.reshape(Train_X.shape[0]*9, r, w, 3)
			Test_X_spatial = Test_X.reshape(Test_X.shape[0]* 9, r, w, 3)
			# Extend Y labels 10 fold, so that all images have labels
			Train_Y_spatial = np.repeat(Train_Y, 9, axis=0)
			Test_Y_spatial = np.repeat(Test_Y, 9, axis=0)
		
		elif dB == 'CASME2_RGB':
			Train_X_spatial = Train_X.reshape(Train_X.shape[0]*10, r, w, 3)
			Test_X_spatial = Test_X.reshape(Test_X.shape[0]* 10, r, w, 3)
			# Extend Y labels 10 fold, so that all images have labels
			Train_Y_spatial = np.repeat(Train_Y,10, axis=0)
			Test_Y_spatial = np.repeat(Test_Y, 10, axis=0)
		

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
			for layer in vgg_model_cam.layers[:31]:
				layer.trainable = False

		if train_spatial_flag == 1 and train_temporal_flag == 1:
			# trains encoder until fc, train temporal
			
			# Spatial Training
			if tensorboard_flag == 1:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
			else:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)

			vgg_model.save_weights(spatial_weights_name + str(sub) + ".h5")
			model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
			plot_model(model, to_file="spatial_module_FULL_TRAINING.png", show_shapes=True)	

			# Spatial Encoding
			output = model.predict(X, batch_size = batch_size)
			features = output.reshape(int(Train_X.shape[0]), timesteps_TIM, output.shape[1])
			
			# Temporal Training
			if tensorboard_flag == 1:
				temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs, callbacks=[tbCallBack])
			else:
				temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs)	

			temporal_model.save_weights(temporal_weights_name + str(sub) + ".h5")

			# Testing
			output = model.predict(test_X, batch_size = batch_size)
			features = output.reshape(int(Test_X.shape[0]), timesteps_TIM, output.shape[1])
			predict = temporal_model.predict_classes(features, batch_size=batch_size)


		elif train_spatial_flag == 1 and train_temporal_flag == 0 and cam_visualizer_flag == 0:
			# trains spatial module ONLY, no escape
			
			# image_generator.fit(X)
			# vgg_model.fit_generator(image_generator.flow(X, y, batch_size=batch_size,
			#  save_to_dir="./augmented/", save_format='png', save_prefix='augmented_me'),
			#   steps_per_epoch=len(X)/batch_size, epochs=spatial_epochs)
			
			# Spatial Training
			if tensorboard_flag == 1:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
			else:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)

			vgg_model.save_weights(spatial_weights_name + str(sub) + ".h5")
			plot_model(vgg_model, to_file="spatial_module_ONLY.png", show_shapes=True)

			# Testing
			predict = vgg_model.predict_classes(test_X, batch_size = batch_size)
			Test_Y_gt = np.repeat(Test_Y_gt, timesteps_TIM, axis=0)

		elif train_spatial_flag == 0 and train_temporal_flag == 1:
			# trains temporal module ONLY.

			# Temporal Training
			if tensorboard_flag == 1:
				temporal_model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=temporal_epochs, callbacks=[tbCallBack])
			else:
				temporal_model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=temporal_epochs)	
				# temporal_model.train_on_batch(Train_X, Train_Y)
			temporal_model.save_weights(temporal_weights_name + str(sub) + ".h5")

			# Testing
			predict = temporal_model.predict_classes(Test_X, batch_size = batch_size)

		elif svm_flag == 1 and finetuning_flag == 0:
			# no finetuning

			X = vgg_model.predict(X, batch_size=batch_size)
			y_for_svm = np.argmax(y, axis=1)

			svm_classifier.fit(X, y_for_svm)

			test_X = vgg_model.predict(test_X, batch_size=batch_size)
			predict = svm_classifier.predict(test_X)

			Test_Y_gt = np.repeat(Test_Y_gt, timesteps_TIM, axis=0)

		elif train_spatial_flag == 1 and train_temporal_flag == 0 and cam_visualizer_flag == 1:
			# trains spatial module & CAM ONLY
			
			# modify model for CAM
			vgg_model_cam = modify_cam(vgg_model_cam)
			vgg_model_cam.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = [metrics.categorical_accuracy])


			# Spatial Training
			if tensorboard_flag == 1:
				vgg_model_cam.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
			else:
				vgg_model_cam.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)

			vgg_model_cam.save_weights(spatial_weights_name + str(sub) + ".h5")
			plot_model(vgg_model_cam, to_file="spatial_module_CAM_ONLY.png", show_shapes=True)

			# Testing
			predict = vgg_model_cam.predict_classes(test_X, batch_size = batch_size)		


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


		file = open(workplace+'Classification/'+ 'Result/'+dB+'/f1_' + str(train_id) +  '.txt', 'a')
		file.write(str(f1) + "\n")
		file.close()
		##################################################################

		################# write each CT of each CV into .txt file #####################
		record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects)
		###############################################################################