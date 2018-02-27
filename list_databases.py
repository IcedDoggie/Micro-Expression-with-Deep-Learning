import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import pandas as pd

from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, loading_smic_table, loading_casme_table, ignore_casme_samples, ignore_casmergb_samples, LossHistory
from utilities import loading_samm_table

def load_db(db_path, db_name, spatial_size):
	db_home = db_path + db_name  + "/"
	db_images = db_path + db_name + "/" + db_name + "/"

	if db_name == 'CASME2_TIM':
		table = loading_casme_table(db_home + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_images)

		r = w = spatial_size
		subjects=26
		samples = 246
		n_exp = 5
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)

		timesteps_TIM = 10
		data_dim = r * w
		channel = 3

		os.remove(db_home + "Classification/CASME2_TIM_label.txt")

	elif db_name == 'CASME2_Optical':
		table = loading_casme_table(db_home + 'CASME2_label_Ver_2.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_images)

		r = w = spatial_size
		subjects=26
		samples = 246
		n_exp = 5
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 3		

		os.remove(db_home + "Classification/CASME2_Optical_label.txt")

	elif db_name == 'SMIC_TIM10':
		table = loading_smic_table(db_path, db_name)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		r = w = spatial_size
		subjects = 16
		samples = 164
		n_exp = 3
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		channel = 3

	elif db_name == 'SAMM_Optical':
		table, table_objective = loading_samm_table(db_path, db_name)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 9
		data_dim = r * w
		channel = 3

	elif db_name == 'SAMM_TIM10':
		table, table_objective = loading_samm_table(db_path, db_name)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		channel = 3
		#########################################################		



	return r, w, subjects, samples, n_exp, VidPerSubject, timesteps_TIM, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples, db_home, db_images

def restructure_data(channel_flag, subject, subperdb, labelpersub, subjects, n_exp, r, w, timesteps_TIM, channel):
	Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(subject, subperdb, labelpersub, subjects, n_exp)
	# Rearrange Training labels into a vector of images, breaking sequence
	Train_X_spatial = Train_X.reshape(Train_X.shape[0]*timesteps_TIM, r, w, channel)
	Test_X_spatial = Test_X.reshape(Test_X.shape[0]* timesteps_TIM, r, w, channel)

	# Extend Y labels 10 fold, so that all images have labels
	Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
	Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)		

		
	if channel == 1:
		# Duplicate channel of input image
		Train_X_spatial = duplicate_channel(Train_X_spatial)
		Test_X_spatial = duplicate_channel(Test_X_spatial)

	X = Train_X_spatial.reshape(Train_X_spatial.shape[0], channel, r, w)
	y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], n_exp)
	normalized_X = X.astype('float32') / 255.

	test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], channel, r, w)
	test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], n_exp)
	normalized_test_X = test_X.astype('float32') / 255.

	return Train_X, Train_Y, Test_Y, Test_Y, Test_Y_gt, X, y, test_X, test_y