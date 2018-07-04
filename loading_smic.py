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
from utilities import record_scores
from models import VGG_16

def loading_labels(root_db_path, dB, databaseType):

	label_filename = "SMIC_label.xlsx"

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path)
	label_file = label_file.dropna()

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	label = label_file[['Label']]
	num_frames = label_file[['Frames']]

	# print(label_file)
	return subject, filename, label, num_frames

def Read_SMIC_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size):
	r = w = spatial_size
	SubperdB = []

	for sub in sorted([ int(infile[1:]) for infile in os.listdir(inputDir) ]):

		sub_str = "s" + str(sub)

		VidperSub = []

		for type_me in sorted([ inrfile for inrfile in os.listdir(inputDir + sub_str + "/micro/") ]):
			
			for vids in sorted([ video for video in os.listdir(inputDir + sub_str + "/micro/" + type_me + "/") ]):
				
				first_frame = True
				
				for item in sorted([image for image in os.listdir(inputDir + sub_str + "/micro/" + type_me + "/" + vids + "/")]):
					item = inputDir + sub_str + "/micro/" + type_me + "/" + vids + "/" + item
					img = cv2.imread(item)
					[_, _, dim] = img.shape

					if dim == 3:
						img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

					if resizedFlag == 1:
						img = cv2.resize(img, (r, w))

					if first_frame:
						FrameperVid = img.flatten()
					else:
						FrameperVid = np.vstack(( FrameperVid, img.flatten() ))

					first_frame = False

				VidperSub.append(FrameperVid)

				# the label in xcel is not the same as in the path
				if sub < 10:
					vids = vids[:1] + "0" + vids[1:]



				collectinglabel(table, sub, vids, workplace + "Classification/", dB)

		SubperdB.append(VidperSub)		

	return SubperdB

############################## Loading Labels & Images ##############################
root_db_path = "/media/ice/OS/Datasets/"
dB = "SMIC"
databaseType = ["HS", "NIR", "VIS"]
inputDir = root_db_path + dB + "/SMIC_all_cropped/" + databaseType[0] + "/"
workplace = root_db_path + dB + "/"


subject, filename, label, num_frames = loading_labels(root_db_path, dB, databaseType)
filename = filename.as_matrix()
label = label.as_matrix()

table = np.transpose( np.array( [filename, label] ) )	

# os.remove(workplace + "Classification/SMIC_label.txt")

################# Variables #############################
spatial_size = 224
r = w = spatial_size
subjects=26
# subjects=2
samples = 246
n_exp = 5

IgnoredSamples_index = np.empty([0])
VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
listOfIgnoredSamples = []

timesteps_TIM = 10
data_dim = r * w
pad_sequence = 10

#########################################################


############## Flags ####################
resizedFlag = 1
train_spatial_flag = 0
train_temporal_flag = 0
svm_flag = 1
finetuning_flag = 0
tensorboard_flag = 0
#########################################

############## Reading Images and Labels ################
SubperdB = Read_SMIC_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size)
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
