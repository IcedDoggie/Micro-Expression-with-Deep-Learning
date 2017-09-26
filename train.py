import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd

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
from keras.layers import Conv2D

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching

############## Path Preparation ######################
dB = "CASME2_TIM"
workplace = '/media/ice/OS/Datasets/' + dB + "/"
inputDir = '/media/ice/OS/Datasets/' + dB + "/" + dB + "/" 
######################################################

############# Reading Labels from XCEL ########################
wb=xlrd.open_workbook('/media/ice/OS/Datasets/CASME2_label_Ver_2.xls')
ws=wb.sheet_by_index(0)    
colm=ws.col_slice(colx=0,start_rowx=1,end_rowx=None)
iD=[str(x.value) for x in colm]
colm=ws.col_slice(colx=1,start_rowx=1,end_rowx=None)
vidName=[str(x.value) for x in colm]
colm=ws.col_slice(colx=6,start_rowx=1,end_rowx=None)
expression=[str(x.value) for x in colm]
table=np.transpose(np.array([np.array(iD),np.array(vidName),np.array(expression)],dtype=str))
###############################################################

############## Variables ###################
r=50; w=50
resizedFlag=1
subjects=26
samples=246
n_exp=5
VidPerSubject = get_subfolders_num(inputDir)
############################################

###################### Samples to-be ignored ##########################
IgnoredSamples = ['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
					'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
					'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/']
listOfIgnoredSamples=[]
for s in range(len(IgnoredSamples)):
	if s==0:
		listOfIgnoredSamples=[inputDir+IgnoredSamples[s]]
	else:
		listOfIgnoredSamples.append(inputDir+IgnoredSamples[s])
#######################################################################

############ Reading Images and Labels ################
SubperdB = Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace)
print("Loaded Images into the tray...")
labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
print("Loaded Labels into the tray...")
#######################################################


########### Model #######################
model = Sequential()
model.add(Conv2D( 256, kernel_size=(3, 3), strides=(1,1), input_shape=(50, 50, 1) ))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=[metrics.categorical_accuracy])
#########################################

########### Training Process ############
# Todo:
# 1) LOSO (done)
# 2) call model
# 3) saving model architecture
# 4) Saving Checkpoint
# 5) make prediction

for sub in range(subjects):
	Train_X, Train_Y, Test_X, Test_Y = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects)
	
	# Train_X	= Train_X.reshape(2370, 1, 50, 50)

	# history_callback = model.fit(Train_X, Train_Y, validation_split=0.05, epochs=1, batch_size=1)


