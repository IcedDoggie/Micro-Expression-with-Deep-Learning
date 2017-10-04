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

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

def Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size):
	# r=224; w=224
	r=w=spatial_size	
	SubperdB=[]

	for sub in sorted([infile for infile in os.listdir(inputDir)]):
			VidperSub=[]        

			for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
				
				path=inputDir + sub + '/'+ vid + '/'
				if path in listOfIgnoredSamples:
					continue
				# print(dB)
				# print(path)
				imgList=readinput(path,dB)
			  
				numFrame=len(imgList)

				if resizedFlag ==1:
					col=w
					row=r
				else:
					img=cv2.imread(imgList[0])
					[row,col,_l]=img.shape
	##            ##read the label for each input video

				collectinglabel(table, sub[3:], vid, workplace+'Classification/', dB)

				for var in range(numFrame):
					img=cv2.imread(imgList[var])
					[_,_,dim]=img.shape
					
					if dim ==3:

						img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

					if resizedFlag ==1:
						#in resize function, [col,row]
						img=cv2.resize(img,(col,row))
						
			
					if var==0:
						FrameperVid=img.flatten()
					else:
						FrameperVid=np.vstack((FrameperVid,img.flatten()))
					
				
				VidperSub.append(FrameperVid)       

			SubperdB.append(VidperSub)	
	return SubperdB

def label_matching(workplace, dB, subjects, VidPerSubject):
	label=np.loadtxt(workplace+'Classification/'+ dB +'_label.txt')
	labelperSub=[]
	counter = 0
	for sub in range(subjects):
		numVid=VidPerSubject[sub]
		labelperSub.append(label[counter:counter+numVid])
		counter = counter + numVid

	return labelperSub

def get_subfolders_num(path, IgnoredSamples_index):
	files = folders = 0
	# print(path)
	folders_array = np.empty([0])
	subject_array = np.empty([0])

	for root, dirnames, filenames in os.walk(path):
		files += len(filenames)
		folders += len(dirnames)


		if len(dirnames) > 0:
			folders_array = np.append(folders_array, len(dirnames))
	# print(type(folders_array[0]))
	# folders -= 26 # hardcoded, because it includes the root path
	folders_array = np.delete(folders_array, [0]) # remove first element as it includes number of folders from root path
	
	####### Minus out the ignored samples ############
	# print(folders_array)
	for item in IgnoredSamples_index:
		item = int(item)
		folders_array[item] -= 1
	# print(folders_array)
	##################################################

	folders_array = folders_array.tolist()
	folders_array = [int(i) for i in folders_array]
	print( "{:,} files, {:,} folders".format(files, folders) )
	return folders_array


def data_loader_with_LOSO(subject, SubjectPerDatabase, y_labels, subjects):
	Train_X = []
	Train_Y = []
	Test_X = np.array(SubjectPerDatabase[subject])
	Test_Y = np_utils.to_categorical(y_labels[subject], 5)

	########### Leave-One-Subject-Out ###############
	if subject==0:
		for i in range(1,subjects):
			Train_X.append(SubjectPerDatabase[i])
			Train_Y.append(y_labels[i])
	   
	elif subject==subjects-1:
		for i in range(subjects-1):
			Train_X.append(SubjectPerDatabase[i])
			Train_Y.append(y_labels[i])
	   
	else:
		for i in range(subjects):
			if subject == i:
				continue
			else:
				Train_X.append(SubjectPerDatabase[i])
				Train_Y.append(y_labels[i])	

	##################################################

	############ Conversion to numpy and stacking ##############
	Train_X=np.vstack(Train_X) 
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y,5)
	#############################################################
	# print ("Train_X_shape: " + str(np.shape(Train_X)))
	# print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	# print ("Test_X_shape: " + str(np.shape(Test_X)))	
	# print ("Test_Y_shape: " + str(np.shape(Test_Y)))	

	return Train_X, Train_Y, Test_X, Test_Y


def duplicate_channel(X):

	X = np.repeat(X, 3, axis=3)
	# np.set_printoptions(threshold=np.nan)
	# print(X)
	print(X.shape)

	return X