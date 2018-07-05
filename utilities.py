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
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio


from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
import keras
import pydot, graphviz
from keras.utils import np_utils, plot_model

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
import itertools
from pynvml.pynvml import *



def Read_Input_Images(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, spatial_size, channel, objective_flag):
	r = w = spatial_size	
	SubperdB = []

	# cross-checking parameter
	subperdb_id = []

	for sub in sorted([infile for infile in os.listdir(inputDir)]):
		VidperSub = [] 
		vid_id = np.empty([0])       

		for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
				
			path = inputDir + sub + '/' + vid + '/' # image loading path
			if path in listOfIgnoredSamples:
				continue

			imgList = readinput(path)  
			numFrame = len(imgList)

			if resizedFlag == 1:
				col = w
				row = r
			else:
				img = cv2.imread(imgList[0])
				[row,col,_l] = img.shape

			## read the label for each input video
			collectinglabel(table, sub[3:], vid, workplace+'Classification/', dB, objective_flag)


			for var in range(numFrame):
				img = cv2.imread(imgList[var])
					
				[_,_,dim] = img.shape
					
				if channel == 1:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				if resizedFlag == 1:
					img = cv2.resize(img, (col,row))
						
			
				if var == 0:
					FrameperVid = img.flatten()
				else:
					FrameperVid = np.vstack((FrameperVid,img.flatten()))
					
				vid_id = np.append(vid_id, imgList[var]) # <--cross-check
			VidperSub.append(FrameperVid)       
	
		subperdb_id.append(vid_id)# <--cross-check
		SubperdB.append(VidperSub)	

	return SubperdB

def label_matching(workplace, dB, subjects, VidPerSubject):
	label=np.loadtxt(workplace+'Classification/'+ dB +'_label.txt')
	labelperSub=[]
	counter = 0
	for sub in range(subjects):
		# print(sub)
		numVid=VidPerSubject[sub]
		labelperSub.append(label[counter:counter+numVid])
		counter = counter + numVid

	return labelperSub


def get_vid_per_subject(table, listOfIgnoredLabels):
	pdt = pd.DataFrame(data=table[0:,0:],columns=['sub','id','y'])
	pdt = pdt[~(pdt['y'].isin(listOfIgnoredLabels))]

	out = pdt.groupby('sub').size().tolist()
	return out




def get_subfolders_num(path, IgnoredSamples_index):
	files = folders = 0
	# print(path)
	folders_array = np.empty([0])
	subject_array = np.empty([0])

	for root, dirnames, filenames in os.walk(path):
		files += len(filenames)
		folders += len(dirnames)


		if len(dirnames) > 0:
			# print(dirnames)
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
	# print( "{:,} files, {:,} folders".format(files, folders) )
	return folders_array

def standard_data_loader(SubjectPerDatabase, y_labels, subjects, classes):
	Train_X = []
	Train_Y = []
	Test_Y_gt = np.empty([0])
	for subject in range((subjects)):

		Train_X.append(SubjectPerDatabase[subject])
		Train_Y.append(y_labels[subject])
		Test_Y_gt = np.append(Test_Y_gt, y_labels[subject])
	# print(Train_Y)

	# print(Test_Y_gt)
	############ Conversion to numpy and stacking ###############
	Train_X=np.vstack(Train_X)
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y, classes)
	#############################################################
	# print ("Train_X_shape: " + str(np.shape(Train_X)))
	# print ("Train_Y_shape: " + str(np.shape(Train_Y)))


	return Train_X, Train_Y, Test_Y_gt


def data_loader_with_LOSO(subject, SubjectPerDatabase, y_labels, subjects, classes):
	Train_X = []
	Train_Y = []


	Test_X = np.array(SubjectPerDatabase[subject])
	Test_Y = np_utils.to_categorical(y_labels[subject], classes)
	Test_Y_gt = y_labels[subject]

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


	############ Conversion to numpy and stacking ###############
	Train_X=np.vstack(Train_X)
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y, classes)
	#############################################################

	return Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt


def duplicate_channel(X):

	X = np.repeat(X, 3, axis=3)
	# np.set_printoptions(threshold=np.nan)
	# print(X)
	print(X.shape)

	return X

def record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects):
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

def loading_smic_labels(root_db_path, dB):

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

def loading_samm_labels(root_db_path, dB, objective_flag):
	label_filename = 'SAMM_Micro_FACS_Codes_v2.xlsx'

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path, converters={'Subject': lambda x: str(x)})
	# remove class 6, 7
	if objective_flag:
		print(objective_flag)
		label_file = label_file.ix[label_file['Objective Classes'] < 6]

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	label = label_file[['Estimated Emotion']]
	objective_classes = label_file[['Objective Classes']]
	# print(label)



	return subject, filename, label, objective_classes

def loading_casme_labels(root_db_path, dB):
	label_filename = 'CASME2-ObjectiveClasses.xlsx'

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path, converters={'Subject': lambda x: str(x)})

	# remove class 6, 7
	label_file = label_file.ix[label_file['Objective Class'] < 6]
	# print(len(label_file)) # 185 samples

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	objective_classes = label_file[['Objective Class']]

	return subject, filename, objective_classes


def loading_casme_objective_table(root_db_path, dB):
	subject, filename, objective_classes = loading_casme_labels(root_db_path, dB)
	
	subject = subject.as_matrix()
	filename = filename.as_matrix()
	objective_classes = objective_classes.as_matrix()

	table = np.transpose( np.array( [subject, filename, objective_classes] ) )

	return table



def loading_casme_table(xcel_path):
	wb=xlrd.open_workbook(xcel_path)
	ws=wb.sheet_by_index(0)    
	colm=ws.col_slice(colx=0,start_rowx=1,end_rowx=None)
	iD=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=1,start_rowx=1,end_rowx=None)
	vidName=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=6,start_rowx=1,end_rowx=None)
	expression=[str(x.value) for x in colm]
	table=np.transpose(np.array([np.array(iD),np.array(vidName),np.array(expression)],dtype=str))	
	# print(table)
	return table


def loading_smic_table(root_db_path, dB):
	subject, filename, label, num_frames = loading_smic_labels(root_db_path, dB)
	filename = filename.as_matrix()
	label = label.as_matrix()

	table = np.transpose( np.array( [filename, label] ) )	
	return table	


def loading_samm_table(root_db_path, dB, objective_flag):	
	subject, filename, label, objective_classes = loading_samm_labels(root_db_path, dB, objective_flag)
	# print("subject:%s filename:%s label:%s objective_classes:%s" %(subject, filename, label, objective_classes))
	subject = subject.as_matrix()
	filename = filename.as_matrix()
	label = label.as_matrix()
	objective_classes = objective_classes.as_matrix()
	table = np.transpose( np.array( [filename, label] ) )
	table_objective = np.transpose( np.array( [subject, filename, objective_classes] ) )
	# print(table)
	return table, table_objective

def filter_objective_samples(table): # this is to filter data with objective classes which is 1-5, omitting 6 and 7
	list_samples = []
	sub = table[0, :, 0]
	vid = table[0, :, 1]
	# print(sub)
	# print(vid)

	for count in range(len(sub)):
		pathname = 0
		if len(sub[count]) == 2:
			pathname = "sub" + sub[count] + "/" + vid[count]
		else:
			pathname = sub[count] + "/" + vid[count]
		# pathname = inputDir + pathname
		list_samples += [pathname]

	# print(list_samples)

	return list_samples

def ignore_casme_samples(db_path, list_db):
	# ignored due to:
	# 1) no matching label.
	# 2) fear, sadness are excluded due to too little data, see CASME2 paper for more
	IgnoredSamples = ['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
						'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
						'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/' ]
	# IgnoredSamples = ['sub09/EP02_02f/', 'sub24/EP02_07/']
	# inputDir2 = "/media/ice/OS/Datasets/" + 'CASME2_Strain_TIM10' + '/' + 'CASME2_Strain_TIM10' + '/'
	# inputDir3 = "/media/ice/OS/Datasets/" + 'CASME2_TIM' + '/' + 'CASME2_TIM' + "/"
	listOfIgnoredSamples=[]
	first_flag = 1
	for s in range(len(IgnoredSamples)):
		for db in list_db:
			ignore_path = db_path + db + "/" + db + "/"

			if s == 0 and first_flag:
				listOfIgnoredSamples=[ignore_path + IgnoredSamples[s]]
				first_flag = 0
			
			else:
				listOfIgnoredSamples.append(ignore_path + IgnoredSamples[s])

	### Get index of samples to be ignored in terms of subject id ###
	IgnoredSamples_index = np.empty([0])
	sub_items = np.empty([0])
	for item in IgnoredSamples:
		sub_item = item.split('/', 1)[0]
		sub_items = np.append(sub_items, sub_item)
		item = item.split('sub', 1)[1]
		# print(sub_item)
		item = int(item.split('/', 1)[0]) - 1 
		IgnoredSamples_index = np.append(IgnoredSamples_index, item)


	return listOfIgnoredSamples, IgnoredSamples_index

def ignore_casmergb_samples(inputDir): # not a universal function, only specific to casme2_tim and derived data from casme2_tim
	# ignored due to:
	# 1) no matching label.
	# 2) fear, sadness are excluded due to too little data, see CASME2 paper for more
	IgnoredSamples = ['sub09/EP02_02f/', 'sub24/EP02_07/']
	# IgnoredSamples = ['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
	# 					'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
	# 					'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/']	 
	listOfIgnoredSamples=[]
	for s in range(len(IgnoredSamples)):
		if s==0:
			listOfIgnoredSamples=[inputDir+IgnoredSamples[s]]
		else:
			listOfIgnoredSamples.append(inputDir+IgnoredSamples[s])
	### Get index of samples to be ignored in terms of subject id ###
	IgnoredSamples_index = np.empty([0])
	for item in IgnoredSamples:
		item = item.split('sub', 1)[1]
		item = int(item.split('/', 1)[0]) - 1 
		IgnoredSamples_index = np.append(IgnoredSamples_index, item)


	return listOfIgnoredSamples, IgnoredSamples_index	

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []
		self.epochs = []
	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracy.append(logs.get('categorical_accuracy'))
		self.epochs.append(logs.get('epochs'))


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def record_loss_accuracy(db_home, train_id, db, history_callback):
	file_loss = open(db_home + 'Classification/' + 'Result/' + db + '/loss_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(history_callback.losses) + "\n")
	file_loss.close()

	file_loss = open(db_home + 'Classification/' + 'Result/' + db + '/accuracy_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(history_callback.accuracy) + "\n")
	file_loss.close()	

	file_loss = open(db_home + 'Classification/' + 'Result/'+ db + '/epoch_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(history_callback.epochs) + "\n")
	file_loss.close()		

def record_weights(model, weights_name, subject, flag):
	model.save_weights(weights_name + str(subject) + ".h5")

	if flag == 's' or flag == 'st':
		model = Model(inputs=model.input, outputs=model.layers[35].output)
		plot_model(model, to_file = "spatial_module_FULL_TRAINING.png", show_shapes=True)	
	else:
		plot_model(model, to_file = "temporal_module.png", show_shapes=True)	

	return model

def sanity_check_image(X, channel, spatial_size):
	# item = X[0,:,:,:]
	item = X[0, :, :, 0]

	item = item.reshape(224, 224, channel)

	cv2.imwrite('sanity_check.png', item)


def gpu_observer():

	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
			nvmlDeviceGetName(handle),
			meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    

# def concatenate_tim():


def visualize_gradcam():
	# visualize cam

	countcam = 0

	for item_idx in range(len(predict)):
		# if predict[item_idx] == Test_Y_gt[item_idx]:
		# 	test_gray = Test_X_gray[item_idx + countcam]

		# 	test_gray_4_channel = duplicate_channel(test_gray, 4)
		# 	test_gray_4_channel = test_gray_4_channel.reshape((224, 224, 4))
		# 	test_gray = duplicate_channel(test_gray, 3)
		# 	test_gray = test_gray.reshape((224, 224, 3))

		# 	item = test_gray


		# 	# output for single class
		# 	cam_output = visualize_cam(model, 29, int(predict[item_idx]), test_gray_4_channel)
		# 	overlaying_cam = overlay(item, cam_output)
		# 	cv2.imwrite(image_path + '_' + str(sub) + '_' + str(table[table_count, 1]) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam0.png', overlaying_cam)


		# 	# output for all class
		# 	# cam_output = visualize_cam(model, 29, 0, item)
		# 	# cam_output2 = visualize_cam(model, 29, 1, item)
		# 	# cam_output3 = visualize_cam(model, 29, 2, item)
		# 	# cam_output4 = visualize_cam(model, 29, 3, item)
		# 	# cam_output5 = visualize_cam(model, 29, 4, item)

		# 	# overlaying_cam = overlay(item, cam_output)
		# 	# overlaying_cam2 = overlay(item, cam_output2)
		# 	# overlaying_cam3 = overlay(item, cam_output3)
		# 	# overlaying_cam4 = overlay(item, cam_output4)
		# 	# overlaying_cam5 = overlay(item, cam_output5)



		# 	# cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam0.png', overlaying_cam)
		# 	# cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam1.png', overlaying_cam2)
		# 	# cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam2.png', overlaying_cam3)
		# 	# cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam3.png', overlaying_cam4)
		# 	# cv2.imwrite(image_path + '_' + str(sub) + '_' + str(item_idx) + '_' + str(predict[item_idx]) + '_' + str(Test_Y_gt[item_idx]) + '_coverlayingcam4.png', overlaying_cam5)

		# countcam += 1

		######## write the log file for megc 2018 ############
		# result_string = str(table[0, table_count, 1])  + ' ' + str(int(Test_Y_gt[item_idx])) + ' ' + str(predict[item_idx]) + '\n' # for objective

		result_string = table[table_count, 1]  + ' ' + str(int(Test_Y_gt[item_idx])) + ' ' + str(predict[item_idx]) + '\n'
		file.write(result_string)
		######################################################
		table_count += 1			

