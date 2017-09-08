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
from imgaug import augmenters as iaa

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

def Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace):
	r=50; w=50	
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

def Label_Matching():
	label=np.loadtxt(workplace+'Classification/'+ dB +'_label.txt')
	labelperSub=[]
	counter = 0
	for sub in range(subjects):
		numVid=VidPerSubject[sub]
		labelperSub.append(label[counter:counter+numVid])
		counter = counter + numVid


def get_subfolders_num(path):
	files = folders = 0
	# print(path)
	folders_array = np.empty([0])
	for root, dirnames, filenames in os.walk(path):
		files += len(filenames)
		folders += len(dirnames) 
		if len(dirnames) > 0:
			folders_array = np.append(folders_array, len(dirnames))

	folders -= 26 # hardcoded, because it includes the root path
	folders_array = folders_array.tolist()
	print(folders_array)
	print( "{:,} files, {:,} folders".format(files, folders) )
	return folders_array