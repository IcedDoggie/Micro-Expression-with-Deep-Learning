import sklearn
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall

def data_loader_LOSO(data, y_labels, subject, list_subjects):
	# lbp dim
	lbp_dim = 177
	class_dim = 5

	# loading data
	test_X = np.empty([0])
	X = np.empty([0])
	test_y = np.empty([0])
	y = np.empty([0])

	#### Load test set ####
	test_X = np.array(data[subject])
	test_y = y_labels[subject]
	# print(len(data))
	# test_y = np_utils.to_categorical(y_labels[subject], 5)
	#######################
	# print(len(data))
	########### Leave-One-Subject-Out ###############
	if subject==0:
		for i in range(1,list_subjects):

			X = np.append(X, data[i])
			y = np.append(y, y_labels[i])
	elif subject==list_subjects-1:
		for i in range(list_subjects-1):
			X = np.append(X, data[i])
			y = np.append(y, y_labels[i])
	else:
		for i in range(list_subjects):
			if subject == i:
				continue
			else:
				X = np.append(X, data[i])
				y = np.append(y, y_labels[i])
	##################################################	

	############ Conversion to numpy and stacking ###############
	X = X.reshape(int(len(X)/lbp_dim), lbp_dim)
	# X = np.transpose(X)
	test_X = test_X.reshape(int(len(test_X)/lbp_dim), lbp_dim)
	# y = y.reshape(int(len(y)/class_dim), class_dim)
	# test_y = test_y.reshape(int(len(test_y)/class_dim), class_dim)

	# X=np.vstack(X)
	# y=np.hstack(y)
	# print(test_X.shape)
	# print(test_y)
	# y = np_utils.to_categorical(y, 4)
	#############################################################
	# print ("Train_X_shape: " + str(np.shape(X)))
	# print ("Train_Y_shape: " + str(np.shape(y)))
	# print ("Test_X_shape: " + str(np.shape(test_X)))	
	# print ("Test_Y_shape: " + str(np.shape(test_y)))

	return X, y, test_X, test_y

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
	# print ("Train_X_shape: " + str(np.shape(Train_X)))
	# print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	# print ("Test_X_shape: " + str(np.shape(Test_X)))	
	# print ("Test_Y_shape: " + str(np.shape(Test_Y)))	

	return Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt



def load_labels_and_LBP_values(file_casme, file_samm, lbp_path):
	# get files that should be ignored due to objective class 6, 7
	file_casme = file_casme
	file_samm = file_samm
	lbp_path = lbp_path

	table_casme = pd.read_excel(file_casme, converters={'Subject': lambda x: str(x)})
	table_samm = pd.read_excel(file_samm, converters={'Subject': lambda x: str(x)})
	# remove class 6, 7
	table_casme = table_casme.ix[table_casme['Objective Class'] < 6]	
	table_samm = table_samm.ix[table_samm['Objective Classes'] < 6]	
	table_casme = table_casme[['Subject', 'Filename', 'Objective Class']]
	table_casme['Subject'] = 'sub' + table_casme['Subject'].astype(str)
	table_samm = table_samm[['Subject', 'Filename', 'Objective Classes']]
	table_casme = table_casme.as_matrix()
	table_samm = table_samm.as_matrix()
	table_casme[:, 2] -= 1
	table_samm[:, 2] -= 1
	table = np.concatenate((table_casme, table_samm)) #filtered table
	# table = table_casme
	# print(table)

	reference_table = np.empty([0])
	help_y = np.empty([0])
	help_x = np.empty([0])
	y_labels = []
	list_data = []
	current_sub = str(table[0, 0]) # initial subject

	for item in range(len(table)):

		if current_sub != str(table[item, 0]) or (item + 1)==len(table):
			# print(current_sub)
			y_labels += [help_y]
			list_data += [help_x]
			current_sub = str(table[item, 0])
			help_y = np.empty([0])
			help_x = np.empty([0])

		item_name = lbp_path + str(table[item, 0]) + '_' + table[item, 1] + '.txt'
		data = np.loadtxt(item_name)
		test_data = sum(data)
		print(test_data)
		
		# normalize the value
		# way 1
		# data =  ( data - min(data) ) / ( max(data) - min(data) )
		# way 2
		data = data * 255
		# data = data.astype(int)

		help_y = np.append(help_y, table[item, 2])
		help_x = np.append(help_x, data)

	# print(len(list_data))
	# print(len(y_labels))

	return list_data, y_labels

# some tunable parameters
subjects = 46 # 46 for samm-casme # 26 for casme
samples = 253 # 253 for samm-casme # 185 casme # 68 samm 
casme_path = "/media/ice/OS/Datasets/SAMM_CASME_Optical/CASME2-ObjectiveClasses.xlsx" 
samm_path = "/media/ice/OS/Datasets/SAMM_CASME_Optical/SAMM_Micro_FACS_Codes_v2.xlsx"
lbp_path = '/home/ice/Documents/Micro-Expression/LBP_features/'
image_path = '/media/ice/OS/Datasets/CASME2_Optical/CASME2_Optical/'
n_exp = 5

# load labels and LBP values
data, y_labels = load_labels_and_LBP_values(casme_path, samm_path, lbp_path)	

# print(data)

# print(data)
tot_mat = np.zeros((n_exp,n_exp))# loading lbptop features


	
for sub in range(subjects):
	# X, y, test_X, test_y = data_loader_with_LOSO(sub, data, y_labels, subjects, 5)
	X, y, test_X, test_y = data_loader_LOSO(data, y_labels, sub, subjects)
	# print(X)
	# clf = svm.SVC(kernel = 'linear', C = 0.1, decision_function_shape='ovr')

	
	# lin_clf = svm.LinearSVC()
	# lin_clf.fit(X, y)	
	clf = svm.SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
	clf.fit(X, y)
	prediction = clf.predict(test_X)
	print(test_y)
	print(prediction)

	ct = confusion_matrix(test_y, prediction)

	# check the order of the CT
	order = np.unique(np.concatenate((prediction,test_y)))
	# create an array to hold the CT for each CV
	mat = np.zeros((n_exp,n_exp))
	# print(mat.shape)
	# print(ct.shape)
	# put the order accordingly, in order to form the overall ConfusionMat
	for m in range(len(order)):
		for n in range(len(order)):
			mat[int(order[m]),int(order[n])]=ct[m,n]
		   
	tot_mat = mat + tot_mat	

	microAcc = np.trace(tot_mat) / np.sum(tot_mat)
	[f1,precision,recall] = fpr(tot_mat,n_exp)

	print("f1: " + str(f1))
print(tot_mat)
war = weighted_average_recall(tot_mat, n_exp, samples)
uar = unweighted_average_recall(tot_mat, n_exp)
print("war: " + str(war))
print("uar: " + str(uar))			