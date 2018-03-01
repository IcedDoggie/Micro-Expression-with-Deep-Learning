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
from vis.visualization import visualize_cam, overlay, visualize_activation


from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, loading_smic_table, loading_casme_table, ignore_casme_samples, ignore_casmergb_samples, LossHistory
from utilities import loading_samm_table, plot_confusion_matrix
from models import VGG_16, temporal_module, VGG_16_4_channels, convolutional_autoencoder


def test_casme(batch_size, spatial_epochs, temporal_epochs, train_id, dB, spatial_size, flag, tensorboard):
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
		subjects=26
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
	elif flag == 'st4vis':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 3	
	#########################################



	############ Reading Images and Labels ################
	SubperdB = Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size, channel)
	print("Loaded Images into the tray...")
	labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
	print("Loaded Labels into the tray...")

	if channel_flag == 1:
		inputDir = root_db_path + dB + "/" + dB + "/" 
		
		SubperdB_strain = Read_Input_Images(root_db_path + 'CASME2_Strain_TIM10' + '/' + 'CASME2_Strain_TIM10' + '/', listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table, workplace, spatial_size, 3)
		SubperdB_gray = Read_Input_Images(root_db_path + 'CASME2_TIM' + '/' + 'CASME2_TIM' + '/', listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, workplace, spatial_size, 3)		

	elif channel_flag == 3:
		inputDir_strain = '/media/ice/OS/Datasets/CASME2_Strain_TIM10/CASME2_Strain_TIM10/'		
		SubperdB_strain = Read_Input_Images(inputDir_strain, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table, workplace, spatial_size, 3)
		inputDir_gray = '/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/'
		SubperdB_gray = Read_Input_Images(inputDir_gray, listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, workplace, spatial_size, 3)		

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
		data_dim = 8192

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



	weights_dir = '/media/ice/OS/Datasets/Weights/53/'
	image_path = '/home/ice/Documents/Micro-Expression/image/'
	table_count = 0
	for sub in range(subjects):
		############### Reinitialization & weights reset of models ########################

		temporal_model_weights = weights_dir + 'temporal_enrichment_ID_' + str(train_id) + '_' + str(dB) + '_' + str(sub) + '.h5'
		vgg_model_weights = weights_dir + 'vgg_spatial_'+ str(train_id) + '_' + str(dB) + '_' + str(sub) + '.h5'
		vgg_model_strain_weights = weights_dir + 'vgg_spatial_strain_'+ str(train_id) + '_' + str(dB) + '_' + str(sub) + '.h5'
		conv_ae_weights = weights_dir  + 'autoencoder_' + str(train_id) + '_' + str(dB) + '_' + str(sub) + '.h5'
		conv_ae_strain_weights = weights_dir + 'autoencoder_strain_' + str(train_id) + '_' + str(dB) + '_' + str(sub) + '.h5'


		temporal_model = temporal_module(data_dim=data_dim, timesteps_TIM=timesteps_TIM, weights_path=temporal_model_weights)
		temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

		conv_ae = convolutional_autoencoder(spatial_size = spatial_size, weights_path=conv_ae_weights)
		conv_ae.compile(loss='binary_crossentropy', optimizer=adam)

		conv_ae_strain = convolutional_autoencoder(spatial_size = spatial_size, weights_path=conv_ae_strain_weights)
		conv_ae_strain.compile(loss='binary_crossentropy', optimizer=adam)


		vgg_model = VGG_16(spatial_size = spatial_size, classes=classes, weights_path=vgg_model_weights)
		vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

		vgg_model_strain = VGG_16(spatial_size = spatial_size, classes=classes, weights_path=vgg_model_strain_weights)
		vgg_model_strain.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])		

		svm_classifier = SVC(kernel='linear', C=1)
		####################################################################################
		
		Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects, classes)

		# Rearrange Training labels into a vector of images, breaking sequence
		Train_X_spatial = Train_X.reshape(Train_X.shape[0]*timesteps_TIM, r, w, channel)
		Test_X_spatial = Test_X.reshape(Test_X.shape[0]* timesteps_TIM, r, w, channel)

		
		# Special Loading for 4-Channel
		if channel_flag == 1 or channel_flag == 3:
			Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects, classes)
			Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0]*timesteps_TIM, r, w, 3)
			Test_X_Strain = Test_X_Strain.reshape(Test_X.shape[0]*timesteps_TIM, r, w, 3)


			Train_X_Gray, _, Test_X_Gray, _, _ = data_loader_with_LOSO(sub, SubperdB_gray, labelperSub, subjects, classes)
			Test_X_Gray = Test_X_Gray.reshape(Test_X_Gray.shape[0]*10, r, w, 3)			
			# print(Train_X_Strain.shape)
			# Train_X_Strain = Train_X_Strain[0]
			# Train_X_Strain = Train_X_Strain.reshape((224, 224, 3, 1))
			# Train_X_Strain = Train_X_Strain.reshape((224, 224, 3))

			# cv2.imwrite('steveharvey.png', Train_X_Strain)
			# Concatenate Train X & Train_X_Strain
			# Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain), axis=3)
			# Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain), axis=3)

			total_channel = 4


		# Extend Y labels 10 fold, so that all images have labels
		Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
		Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)		


		##################### Training & Testing #########################

		# print(Train_X_spatial.shape)	


		test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], channel, r, w)
		test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], classes)
		normalized_test_X = test_X.astype('float32') / 255.



		Test_X_Strain = Test_X_Strain.reshape(Test_X_Strain.shape[0], channel, r, w)
		Test_X_Gray = Test_X_Gray.reshape(Test_X_Gray.shape[0], channel, r, w)
		# test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], classes)
		normalized_test_X_strain = test_X.astype('float32') / 255.

		# print(X.shape)

		###### conv weights must be freezed for transfer learning ######
		if finetuning_flag == 1:
			for layer in vgg_model.layers[:33]:
				layer.trainable = False

		if train_spatial_flag == 1 and train_temporal_flag == 1:

			# vgg
			model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
			plot_model(model, to_file="spatial_module_FULL_TRAINING.png", show_shapes=True)	
			output = model.predict(test_X)


			# vgg strain
			model_strain = Model(inputs=vgg_model_strain.input, outputs=vgg_model_strain.layers[35].output)
			plot_model(model_strain, to_file="spatial_module_FULL_TRAINING_strain.png", show_shapes=True)	
			output_strain = model_strain.predict(Test_X_Strain)

			# ae 
			# model_ae = Model(inputs=conv_ae.input, outputs=conv_ae.output)
			# plot_model(model_ae, to_file='autoencoders.png', show_shapes=True)			
			# output_ae = model_ae.predict(normalized_test_X)
			# output_ae = model.predict(output_ae)

			# ae strain
			# model_ae_strain = Model(inputs=conv_ae_strain.input, outputs=conv_ae_strain.output)
			# plot_model(model_ae, to_file='autoencoders.png', show_shapes=True)
			# output_ae_strain = model_ae_strain.predict(normalized_test_X_strain)
			# output_ae_strain = model_ae_strain.predict(output_ae_strain)


			# concatenate features
			output = np.concatenate((output, output_strain), axis=1)
			features = output.reshape(int(Test_X.shape[0]), timesteps_TIM, output.shape[1])

			# temporal
			predict = temporal_model.predict_classes(features, batch_size=batch_size)


			# visualize cam
			countcam = 0
			file = open(workplace+'Classification/'+ 'Result/'+dB+'/log_hde' + str(train_id) + '.txt', 'a')
			file.write(str(sub+1) + "\n")			
			for item_idx in range(len(predict)):
				test_strain = Test_X_Gray[item_idx + countcam]
				test_strain = test_strain.reshape((224, 224, 3))
				item = test_strain

				cam_output = visualize_cam(model, 29, 0, item)
				cam_output2 = visualize_cam(model, 29, 1, item)
				cam_output3 = visualize_cam(model, 29, 2, item)
				cam_output4 = visualize_cam(model, 29, 3, item)
				cam_output5 = visualize_cam(model, 29, 4, item)

				overlaying_cam = overlay(item, cam_output)
				overlaying_cam2 = overlay(item, cam_output2)
				overlaying_cam3 = overlay(item, cam_output3)
				overlaying_cam4 = overlay(item, cam_output4)
				overlaying_cam5 = overlay(item, cam_output5)

				cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam0.png', overlaying_cam)
				cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam1.png', overlaying_cam2)
				cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam2.png', overlaying_cam3)
				cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam3.png', overlaying_cam4)
				cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam4.png', overlaying_cam5)

				countcam += 9

				######## write the log file for megc 2018 ############

				result_string = table[table_count, 1]  + ' ' + str(int(Test_Y_gt[item_idx])) + ' ' + str(predict[item_idx]) + '\n'
				file.write(result_string)
				######################################################
				table_count += 1				
		##############################################################

		#################### Confusion Matrix Construction #############
		print (predict)
		print (Test_Y_gt)	

		ct = confusion_matrix(Test_Y_gt,predict)
		# print(type(ct))a
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

	tot_mat_cm = np.asarray(tot_mat, dtype=int)
			
	plt.figure()
	classes_test = [0, 1, 2, 3, 4]
	plot_confusion_matrix(tot_mat_cm, classes_test, normalize=True, title='Confusion matrix_single_db')

	plt.show()