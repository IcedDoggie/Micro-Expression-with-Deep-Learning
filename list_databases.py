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



	return r, w, subjects, samples, n_exp, VidPerSubject, timesteps_TIM, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples