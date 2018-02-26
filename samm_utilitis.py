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
from itertools import groupby


from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio


from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
import keras

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

def Read_Input_Images_SAMM_CASME(inputDir, filteredSamples, ignoredSamples, dB, resizedFlag, table, workplace, spatial_size, channel):
	# r=224; w=224
	r=w=spatial_size	
	SubperdB=[]

	# cross-checking parameter
	
	subperdb_id = []

	for sub in sorted([infile for infile in os.listdir(inputDir)]):
			VidperSub=[] 
			vid_id = np.empty([0])       

			for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
				# /media/ice/OS/Datasets/SAMM_CASME_Optical/SAMM_CASME_Optical/006/006_1_2/
				# /media/ice/OS/Datasets/SAMM_CASME_Optical/SAMM_CASME_Optical/006/006_1_2/
				# /media/ice/OS/Datasets/SAMM_CASME_Optical/SAMM_CASME_Optical/sub09/EP13_02/
				path=inputDir + sub + '/'+ vid + '/'
				# print(len(filteredSamples))
				# print(filteredSamples)
				# print("bohaha")

				# filtered samples are samples needed
				if path not in filteredSamples:
					# print(path)
					continue

						
				# print(dB)
				# print(path)
				imgList=readinput(path,dB)
			  
				numFrame=len(imgList)
				# print(numFrame)
				if resizedFlag ==1:
					col=w
					row=r
				else:
					img=cv2.imread(imgList[0])
					[row,col,_l]=img.shape
	##            ##read the label for each input video
				# print(sub[3:])
				collectinglabel(table, sub, vid, workplace+'Classification/', dB)


				for var in range(numFrame):
					img=cv2.imread(imgList[var])
					
					[_,_,dim]=img.shape
					
					if channel == 1:

						img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

					if resizedFlag ==1:
						#in resize function, [col,row]
						img=cv2.resize(img,(col,row))
						
			
					if var==0:
						FrameperVid=img.flatten()
					else:
						FrameperVid=np.vstack((FrameperVid,img.flatten()))
					
					vid_id = np.append(vid_id, imgList[var]) # <--cross-check

				VidperSub.append(FrameperVid)       
			

			subperdb_id.append(vid_id)# <--cross-check
			# print(subperdb_id)
			# if len(VidperSub) > 0:
			# print(len(VidperSub))
			SubperdB.append(VidperSub)	

	# return SubperdB, vid_id, subperdb_id
	return SubperdB

def get_subfolders_num_crossdb(path, IgnoredSamples_index, sub_items, table, list_samples):
	files = folders = 0
	# print(path)
	folders_array = np.empty([0])
	subject_array = np.empty([0])

	videos_array = np.empty([0])
	for root, dirnames, filenames in os.walk(path):
		discard_objective_flag = 0
		files += len(filenames)
		folders += len(dirnames)

		# this line bypasses np.delete in 163
		if len(dirnames) > 0 and len(subject_array) == 0:
			subject_array = np.append(subject_array, dirnames)

		elif len(dirnames) > 0:
			folders_array = np.append(folders_array, len(dirnames)) 
			videos_array = np.append(videos_array, (dirnames))




	####### discard objective classses 	with 6 and 7 #######
	item_array = []
	for item in list_samples:
		item = item.split('/', 1)[0]
		if item in sub_items:
			sub_items = np.delete(sub_items, 0) 
		else:
			item_array += [item]

		
	folders_array = [len(list(group)) for key, group in groupby(item_array)]



	folders_array = [int(i) for i in folders_array]


	# process list_samples, modify into proper path for later use
	help_list_samples = []
	for item in list_samples:
		item = path + item + "/"
		help_list_samples += [item]
	list_samples = help_list_samples
	# print(list_samples)
	return folders_array, list_samples	

def loading_samm_labels(root_db_path, dB):
	label_filename = 'SAMM_Micro_FACS_Codes_v2.xlsx'

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path, converters={'Subject': lambda x: str(x)})
	# remove class 6, 7
	label_file = label_file.ix[label_file['Objective Classes'] < 6]
	# print(len(label_file)) # 68 samples

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	label = label_file[['Estimated Emotion']]
	objective_classes = label_file[['Objective Classes']]
	# print(label)

	return subject, filename, label, objective_classes

