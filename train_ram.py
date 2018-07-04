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
import gc

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
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import loading_smic_table, loading_casme_table, loading_samm_table, ignore_casme_samples, ignore_casmergb_samples # data loading scripts
from utilities import record_loss_accuracy, record_weights, record_scores, LossHistory # recording scripts
from utilities import sanity_check_image, gpu_observer
from samm_utilitis import get_subfolders_num_crossdb, Read_Input_Images_SAMM_CASME, loading_samm_labels

from list_databases import load_db, restructure_data
from models import VGG_16, temporal_module, VGG_16_4_channels, convolutional_autoencoder
from new_ram import create_glimpse, Recurrent_Attention_Network, location_network

def train_ram(batch_size, spatial_epochs, temporal_epochs, train_id, list_dB, spatial_size, flag, objective_flag, tensorboard):
	############## Path Preparation ######################
	root_db_path = "/media/ice/OS/Datasets/"
	tensorboard_path = "/home/ice/Documents/Micro-Expression/tensorboard/"
	if os.path.isdir(root_db_path + 'Weights/'+ str(train_id) ) == False:
		os.mkdir(root_db_path + 'Weights/'+ str(train_id) )

	######################################################

	############## Variables ###################
	dB = list_dB[0]
	r, w, subjects, samples, n_exp, VidPerSubject, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples, db_home, db_images, cross_db_flag = load_db(root_db_path, list_dB, spatial_size, objective_flag)

	# avoid confusion
	if cross_db_flag == 1:
		list_samples = listOfIgnoredSamples

	# total confusion matrix to be used in the computation of f1 score
	tot_mat = np.zeros((n_exp, n_exp))

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
	elif flag == 'st4se' or flag == 'st4se_cde':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 1
	elif flag == 'st7se' or flag == 'st7se_cde':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 2
	elif flag == 'st4te' or flag == 'st4te_cde':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 3
	elif flag == 'st7te' or flag == 'st7te_cde':
		train_spatial_flag = 1
		train_temporal_flag = 1
		channel_flag = 4				

	#########################################

	############ Reading Images and Labels ################
	if cross_db_flag == 1:
		SubperdB = Read_Input_Images_SAMM_CASME(db_images, list_samples, listOfIgnoredSamples, dB, resizedFlag, table, db_home, spatial_size, channel)
	else:
		SubperdB = Read_Input_Images(db_images, listOfIgnoredSamples, dB, resizedFlag, table, db_home, spatial_size, channel, objective_flag)


	labelperSub = label_matching(db_home, dB, subjects, VidPerSubject)
	print("Loaded Images into the tray...")
	print("Loaded Labels into the tray...")

	if channel_flag == 1:
		aux_db1 = list_dB[1]
		db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"
		if cross_db_flag == 1:
			SubperdB = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 1)
		else:
			SubperdB_strain = Read_Input_Images(db_strain_img, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 1, objective_flag)

	elif channel_flag == 2:	
		aux_db1 = list_dB[1]
		aux_db2 = list_dB[2]
		db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"	
		db_gray_img = root_db_path + aux_db2 + "/" + aux_db2 + "/"
		if cross_db_flag == 1:
			SubperdB_strain = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 1)
			SubperdB_gray = Read_Input_Images_SAMM_CASME(db_gray_img, list_samples, listOfIgnoredSamples, aux_db2, resizedFlag, table, db_home, spatial_size, 1)
		else:
			SubperdB_strain = Read_Input_Images(db_strain_img, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 1, objective_flag)
			SubperdB_gray = Read_Input_Images(db_gray_img, listOfIgnoredSamples, aux_db2, resizedFlag, table, db_home, spatial_size, 1, objective_flag)

	elif channel_flag == 3:
		aux_db1 = list_dB[1]		
		db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"		
		if cross_db_flag == 1:
			SubperdB = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 3)
		else:
			SubperdB_strain = Read_Input_Images(db_strain_img, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 3, objective_flag)
	
	elif channel_flag == 4: 
		aux_db1 = list_dB[1]
		aux_db2 = list_dB[2]		
		db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"	
		db_gray_img = root_db_path + aux_db2 + "/" + aux_db2 + "/"		
		if cross_db_flag == 1:
			SubperdB_strain = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 3)
			SubperdB_gray = Read_Input_Images_SAMM_CASME(db_gray_img, list_samples, listOfIgnoredSamples, aux_db2, resizedFlag, table, db_home, spatial_size, 3)
		else:
			SubperdB_strain = Read_Input_Images(db_strain_img, listOfIgnoredSamples, aux_db1, resizedFlag, table, db_home, spatial_size, 3, objective_flag)
			SubperdB_gray = Read_Input_Images(db_gray_img, listOfIgnoredSamples, aux_db2, resizedFlag, table, db_home, spatial_size, 3, objective_flag)

	
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
	baseline_history = LossHistory()
	action_history = LossHistory()
	for sub in range(subjects):


		spatial_weights_name = root_db_path + 'Weights/'+ str(train_id) + '/vgg_spatial_'+ str(train_id) + '_' + str(dB) + '_'
		spatial_weights_name_strain = root_db_path + 'Weights/' + str(train_id) + '/vgg_spatial_strain_'+ str(train_id) + '_' + str(dB) + '_' 
		spatial_weights_name_gray = root_db_path + 'Weights/' + str(train_id) + '/vgg_spatial_gray_'+ str(train_id) + '_' + str(dB) + '_'

		temporal_weights_name = root_db_path + 'Weights/' + str(train_id) + '/temporal_ID_' + str(train_id) + '_' + str(dB) + '_' 

		ae_weights_name = root_db_path + 'Weights/' + str(train_id) + '/autoencoder_' + str(train_id) + '_' + str(dB) + '_'
		ae_weights_name_strain = root_db_path + 'Weights/' + str(train_id) + '/autoencoder_strain_' + str(train_id) + '_' + str(dB) + '_'


		
		############ for tensorboard ###############
		#############################################

		Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt, X, y, test_X, test_y = restructure_data(sub, SubperdB, labelperSub, subjects, n_exp, r, w, timesteps_TIM, channel)


		# Special Loading for 4-Channel
		if channel_flag == 1:
			_, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub, SubperdB_strain, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 1)
			
			# verify
			# sanity_check_image(Test_X_Strain, 1, spatial_size)

			# Concatenate Train X & Train_X_Strain
			X = np.concatenate((X, Train_X_Strain), axis=1)
			test_X = np.concatenate((test_X, Test_X_Strain), axis=1)

			total_channel = 4

		elif channel_flag == 2:
			_, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub, SubperdB_strain, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 1)

			_, _, _, _, _, Train_X_Gray, Train_Y_Gray, Test_X_Gray, Test_Y_Gray = restructure_data(sub, SubperdB_gray, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 1)

			# Concatenate Train_X_Strain & Train_X & Train_X_gray
			X = np.concatenate((X, Train_X_Strain, Train_X_gray), axis=1)
			test_X = np.concatenate((test_X, Test_X_Strain, Test_X_gray), axis=1)	

			total_channel = 5		
		
		elif channel_flag == 3:
			_, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub, SubperdB_strain, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 3)

		elif channel_flag == 4:
			_, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub, SubperdB_strain, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 3)
			_, _, _, _, _, Train_X_Gray, Train_Y_Gray, Test_X_Gray, Test_Y_Gray = restructure_data(sub, SubperdB_gray, labelperSub, subjects, n_exp, r, w, timesteps_TIM, 3)


		############### Reinitialization & weights reset of models ########################
		lt = np.array((1, 100))
		no_patches = 4
		scale = (224 - lt[0]) / no_patches
		print(lt.shape)
		print(X[0].shape)

		glimpse_seq = []
		for item in X:
			# print(item.shape)
			glimpse_seq_temp = create_glimpse(item, scale, no_patches, lt)
			glimpse_seq += [glimpse_seq_temp]
			# print(glimpse_seq.shape)
		# glimpse_seq = create_glimpse(X[0], scale, no_patches, lt)
		# print(glimpse_seq.shape)
		glimpse_seq = np.asarray(glimpse_seq)
		glimpse_seq = glimpse_seq.reshape(glimpse_seq.shape[0], glimpse_seq.shape[2], glimpse_seq.shape[3], glimpse_seq.shape[4])
		# print(glimpse_seq.shape)
		model = Recurrent_Attention_Network(glimpse_seq, lt, 256, 1, 112, 256, 56)	
		baseline_model = Recurrent_Attention_Network(glimpse_seq, lt, 256, 1, 112, 256, 56)	
		
		####################################################################################
		

		############### check gpu resources ####################
		gpu_observer()
		########################################################

		##################### Training & Testing #########################
		lt = lt.reshape(1, 2)
		lt = np.repeat(lt, glimpse_seq.shape[0], axis=0)
		model.fit([glimpse_seq, lt], [y], callbacks = [action_history], epochs = spatial_epochs)

		ht_model = Model(inputs = model.input, outputs = model.layers[8].output)
		
		out, ht = ht_model.predict([glimpse_seq, lt])
		lt, log_pi = location_network(ht)

		# reward
		p_class = model.predict([glimpse_seq, lt])
		p_class = (p_class == p_class[:].max(axis = 1)[:, None]).astype(int)
		print(p_class)
		reward = K.cast((p_class == y), dtype = 'float32')
		reward = reward.eval()
		print(reward)

		# train baseline with reward as labels
		baseline_model = Model(inputs = baseline_model.input, outputs = baseline_model.layers[10].output)
		baseline_model.compile(loss=['mean_squared_error'], optimizer=adam, metrics=[metrics.categorical_accuracy])

		baseline_model.fit([glimpse_seq, lt], [reward], callbacks = [baseline_history], epochs = spatial_epochs)
		
		# adjusted reward based on baselines
		adjusted_reward = reward - baseline_model.predict([glimpse_seq, lt])
		# reinforce_loss = sum(-log_pi * adjusted_reward, axis = 1)

		# log_pi = log_pi.reshape(2)
		log_pi = np.repeat(log_pi, 5, axis=1)
		log_pi = log_pi.T
		reinforce_loss = np.mean( np.dot(adjusted_reward, -log_pi))
		print(reinforce_loss)


		hybrid_loss = float(action_history.losses[0]) + float(baseline_history.losses[0]) + reinforce_loss
		print(hybrid_loss)
		##############################################################

		#################### Confusion Matrix Construction #############
		# print (predict)
		# print (Test_Y_gt)	

		# ct = confusion_matrix(Test_Y_gt,predict)
		# # check the order of the CT
		# order = np.unique(np.concatenate((predict,Test_Y_gt)))
		
		# # create an array to hold the CT for each CV
		# mat = np.zeros((n_exp,n_exp))
		# # put the order accordingly, in order to form the overall ConfusionMat
		# for m in range(len(order)):
		# 	for n in range(len(order)):
		# 		mat[int(order[m]),int(order[n])]=ct[m,n]
			   
		# tot_mat = mat + tot_mat
		################################################################
		
		#################### cumulative f1 plotting ######################
		# microAcc = np.trace(tot_mat) / np.sum(tot_mat)
		# [f1,precision,recall] = fpr(tot_mat,n_exp)

		# file = open(db_home+'Classification/'+ 'Result/'+dB+'/f1_' + str(train_id) +  '.txt', 'a')
		# file.write(str(f1) + "\n")
		# file.close()
		##################################################################

		################# write each CT of each CV into .txt file #####################
		# record_scores(db_home, dB, ct, sub, order, tot_mat, n_exp, subjects)
		# war = weighted_average_recall(tot_mat, n_exp, samples)
		# uar = unweighted_average_recall(tot_mat, n_exp)
		# print("war: " + str(war))
		# print("uar: " + str(uar))	
		###############################################################################

		################## free memory ####################

		gc.collect()
		###################################################

# python main.py --train='./train_ram.py' --batch_size=30 --dB='CASME2_TIM' --flag='st' --spatial_size=224 --objective_flag=0