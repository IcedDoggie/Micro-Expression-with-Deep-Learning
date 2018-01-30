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
vgg_model = VGG_16('vgg_spatial_ID_12.h5')
# keras_vgg = keras_vgg16(weights='imagenet')

# vgg_model = VGG_16('imagenet')
vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.sparse_categorical_accuracy])
plot_model(vgg_model, to_file='model.png', show_shapes=True)

svm_classifier = SVC(kernel='linear', C=1)
######################################################
tot_mat = np.zeros((n_exp,n_exp))
for sub in range(subjects):
	image_label_mapping = np.empty([0])

	Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects)

	Test_X_spatial = Test_X.reshape(Test_X.shape[0]* 10, r, w, 1)
	Test_Y_spatial = np.repeat(Test_Y, 10, axis=0)

	# Duplicate channel of input image
	Test_X_spatial = duplicate_channel(Test_X_spatial)		

	test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], 3, r, w)
	test_y = np.repeat(Test_Y_gt, 10, axis=0)



	predict = vgg_model.predict_classes(test_X, batch_size=1)
	counter_item = 0
	for item in (predict):

		if item == 0 or item == 4:
			predict[counter_item] = 1
		elif item == 1 or item == 2:
			predict[counter_item] = 0
		else:
			predict[counter_item] = 2

		counter_item += 1

	print(predict)
	print(test_y)


	ct=confusion_matrix(test_y, predict)
	# check the order of the CT
	order=np.unique(np.concatenate((predict,test_y)))
	
	# create an array to hold the CT for each CV
	mat=np.zeros((n_exp,n_exp))
	# put the order accordingly, in order to form the overall ConfusionMat
	for m in range(len(order)):
		for n in range(len(order)):
			mat[int(order[m]),int(order[n])]=ct[m,n]
		   
	tot_mat = mat + tot_mat
	################################################################
	

	#################### cumulative f1 plotting ######################
	microAcc=np.trace(tot_mat)/np.sum(tot_mat)
	[f1,precision,recall]=fpr(tot_mat,n_exp)


	file = open(workplace+'Classification/'+ 'Result/'+dB+'/f1.txt', 'a')
	file.write(str(f1) + "\n")
	file.close()
	##################################################################

	################# write each CT of each CV into .txt file #####################
	if not os.path.exists(workplace+'Classification/'+'Result/'+dB+'/'):
		os.mkdir(workplace+'Classification/'+ 'Result/'+dB+'/')
		
	with open(workplace+'Classification/'+ 'Result/'+dB+'/sub_CT.txt','a') as csvfile:
			thewriter=csv.writer(csvfile, delimiter=' ')
			thewriter.writerow('Sub ' + str(sub+1))
			thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
			for row in ct:
				thewriter.writerow(row)
			thewriter.writerow(order)
			thewriter.writerow('\n')
			
	if sub==subjects-1:
			# compute the accuracy, F1, P and R from the overall CT
			microAcc=np.trace(tot_mat)/np.sum(tot_mat)
			[f1,p,r]=fpr(tot_mat,n_exp)
			print(tot_mat)
			print("F1-Score: " + str(f1))
			# save into a .txt file
			with open(workplace+'Classification/'+ 'Result/'+dB+'/final_CT.txt','w') as csvfile:
				thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
				for row in tot_mat:
					thewriter.writerow(row)
					
				thewriter=csv.writer(csvfile, delimiter=' ')
				thewriter.writerow('micro:' + str(microAcc))
				thewriter.writerow('F1:' + str(f1))
				thewriter.writerow('Precision:' + str(p))
				thewriter.writerow('Recall:' + str(r))		
	###############################################################################		