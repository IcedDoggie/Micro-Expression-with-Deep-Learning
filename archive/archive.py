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
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, LeakyReLU, Reshape, Conv1D
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
import keras
from keras.preprocessing.image import ImageDataGenerator  


from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from models import LossHistory
from utilities import get_subfolders_num



workplace='/media/ice/OS/Datasets/CASME2_TIM/'
dB="CASME2_TIM"
# rootpath = '/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/'



if dB == "CASME2_raw":
	inputDir='/media/ice/OS/Datasets/CASME2-RAW/'
	resizedFlag=1;

elif dB== "CASME2_large":
	inputDir='/media/ice/OS/Datasets/CASME 2/'
	wb=xlrd.open_workbook('/media/ice/OS/Datasets/CASME 2/CASME2_label_Ver_2.xls');
	ws=wb.sheet_by_index(0)    
	colm=ws.col_slice(colx=0,start_rowx=1,end_rowx=None)
	iD=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=1,start_rowx=1,end_rowx=None)
	vidName=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=6,start_rowx=1,end_rowx=None)
	expression=[str(x.value) for x in colm]
	table=np.transpose(np.array([np.array(iD),np.array(vidName),np.array(expression)],dtype=str))
		
	subjects=26
	samples=246
	n_exp=5
	resizedFlag=1;
	r=68; w=56
	VidPerSubject = [9,13,7,5,19,5,9,3,13,13,10,12,8,4,3,4,34,3,15,11,2,2,12,7,7,16]
	IgnoredSamples=['sub09/EP13_02','sub09/EP02_02f','sub10/EP13_01','sub17/EP15_01',
					'sub17/EP15_03','sub19/EP19_04','sub24/EP10_03','sub24/EP07_01',
					'sub24/EP07_04f','sub24/EP02_07','sub26/EP15_01']
	listOfIgnoredSamples=[]
	for s in range(len(IgnoredSamples)):
		if s==0:
			listOfIgnoredSamples=[inputDir+IgnoredSamples[s]]
		else:
			listOfIgnoredSamples.append(inputDir+IgnoredSamples[s])


elif dB== "CASME2_TIM":
	# Pandas Way
	inputDir='/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/' 
	# excel_file = '/media/ice/OS/Datasets/CASME2_label_Ver_2.xls' 
	# all_data = pd.read_excel(excel_file)

	# iD = all_data[['Subject']]
	# vidName = all_data[['Filename']]
	# expression = all_data[['Estimated Emotion']]
	# table = pd.concat([iD, vidName, expression], axis=1)
	# table = table.as_matrix()
	# print(table)

	# Numpy Way
	wb=xlrd.open_workbook('/media/ice/OS/Datasets/CASME2_label_Ver_2.xls')
	ws=wb.sheet_by_index(0)    
	colm=ws.col_slice(colx=0,start_rowx=1,end_rowx=None)
	iD=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=1,start_rowx=1,end_rowx=None)
	vidName=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=6,start_rowx=1,end_rowx=None)
	expression=[str(x.value) for x in colm]
	table=np.transpose(np.array([np.array(iD),np.array(vidName),np.array(expression)],dtype=str))
	# print(table)
	
	r=50; w=50
	resizedFlag=1;
	subjects=26
	samples=246
	n_exp=5
	VidPerSubject = get_subfolders_num(inputDir)
	# VidPerSubject = [9,13,7,5,19,5,9,3,13,13,10,12,8,4,3,4,34,3,15,11,2,2,12,7,7,16]
	IgnoredSamples=['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
					'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
					'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/']
	listOfIgnoredSamples=[]
	for s in range(len(IgnoredSamples)):
		if s==0:
			listOfIgnoredSamples=[inputDir+IgnoredSamples[s]]
		else:
			listOfIgnoredSamples.append(inputDir+IgnoredSamples[s])

elif dB == "SMIC":
	inputDir="/srv/oyh/DataBase/SMIC/HS_naming_modified/"
	inputDir="/media/ice/OS/Datasets/SMIC"
	wb=xlrd.open_workbook('/srv/oyh/DataBase/SMIC_label.xlsx');
	ws=wb.sheet_by_index(0)    
	colm=ws.col_slice(colx=1,start_rowx=1,end_rowx=None)
	vidName=[str(x.value) for x in colm]
	colm=ws.col_slice(colx=2,start_rowx=1,end_rowx=None)
	expression=[int(x.value) for x in colm]
	table=np.transpose(np.array([np.array(vidName),np.array(expression)],dtype=str))
  
	samples=164; #6 samples are excluded 
	subjects=16;
	n_exp=3;
	r= 170;w=140;
	VidPerSubject = [6,6,39,19,2,4,13,4,7,9,10,10,4,7,2,22];
	listOfIgnoredSamples=[];
	resizedFlag=1;
   

else:
	print("NOT in the selection.")


######### Reading in the input images ########
SubperdB=[]
for sub in sorted([infile for infile in os.listdir(inputDir)]):
		VidperSub=[]        

		for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
			
			path=inputDir + sub + '/'+ vid + '/'
			if path in listOfIgnoredSamples:
				continue

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
					# pass
					img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

				if resizedFlag ==1:
					# in resize function, [col,row]
					img=cv2.resize(img,(col,row))

		
				if var==0:
					FrameperVid=img.flatten()
				else:
					FrameperVid=np.vstack((FrameperVid,img.flatten()))
				
			VidperSub.append(FrameperVid)       

		SubperdB.append(VidperSub)


##### Setting up the LSTM model ########
## Temporal input
data_dim=r*w # 2500
timesteps=10

## Spatial input
image_dim = 280
ndim = 3
batch = 16

model=Sequential()
model.add(LSTM(2000, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(500,return_sequences=False))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(5,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=[metrics.categorical_accuracy])

################ CNN for spatial features ######################
# model_cnn = Sequential()
# model_cnn.add(Conv2D(256, kernel_size = (3, 3), input_shape=((50, 50), 1), data_format=None))
# model_cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=[metrics.categorical_accuracy])
################################################################


######## generate the label based on subjects #########
label = np.loadtxt(workplace+'Classification/'+ dB +'_label.txt')
labelperSub=[]
counter = 0
for sub in range(subjects):
	numVid=VidPerSubject[sub]
	labelperSub.append(label[counter:counter+numVid])
	counter = counter + numVid
######### Image Data Augmentation ############
# datagen = ImageDataGenerator(
	# horizontal_flip = True,
	# rotation_range = 10,
	# rescale = 1.1,
	# )


######## Seperating the input files into LOSO CV ########
tot_mat=np.zeros((n_exp,n_exp))
for sub in range(subjects):
	Train_X=[]
	Train_Y=[]

	Test_X=SubperdB[sub]
	Test_X=np.array(Test_X)
	Test_Y=labelperSub[sub]
	Test_Yy=np_utils.to_categorical(Test_Y, 5)

	##### Leave One Subject Out #######
	if sub==0:
		for i in range(1,subjects):
			Train_X.append(SubperdB[i])
			# print(len(Train_X))
			Train_Y.append(labelperSub[i])
	   
	elif sub==subjects-1:
		for i in range(subjects-1):
			Train_X.append(SubperdB[i])
			Train_Y.append(labelperSub[i])
	   
	else:
		for i in range(subjects):
			if sub == i:
				continue
			else:
				Train_X.append(SubperdB[i])
				Train_Y.append(labelperSub[i])
				
	Train_X=np.vstack(Train_X) 
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y,5)

	print (np.shape(Train_Y))
	print (np.shape(Train_X))
	print (np.shape(Test_Y))	
	print (np.shape(Test_X))

	Train_X_cnn = Train_X[0]
	Train_X_cnn = Train_X_cnn[0]
	print (Train_X_cnn.shape)
	Train_X_cnn = Train_X_cnn.reshape(1, r, w)
	print (Train_X_cnn.shape)
	print (Train_X_cnn.shape[1:])
	Train_X_cnn = Train_X.reshape(2370, 50, 50, 1)
	print (Train_X_cnn.shape)

	# datagen.fit(Train_X_cnn)
	# # flow
	# for X_batch in datagen.flow(Train_X_cnn, batch_size=10, save_to_dir='/media/ice/OS/Datasets/CASME2_TIM/augment', save_prefix='aug', save_format='jpg'):
	# 	print(X_batch.shape)
	# 	for i in range(0, 9):

	# 		plt.subplot(330 + 1 + i)
	# 		plt.imshow(X_batch[i].reshape(50, 50), cmap=plt.get_cmap('gray'))
	# 	plt.show()


	# flow_from_directory

	# train_generator = datagen.flow_from_directory(
	# 	'/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/', 
	# 	target_size = (280, 280),
	# 	class_mode = 'binary',
	# 	save_to_dir = '/media/ice/OS/Datasets/CASME2_TIM/augment/',
	# 	save_prefix = 'augmented',
	# 	save_format = 'jpeg',
	# 	batch_size = 23,
	# 	# seed = ,
	# 	)

	# model.fit_generator(train_generator, steps_per_epoch = 23, epochs=1)	

	history_callback = model.fit(Train_X, Train_Y, validation_split=0.05, epochs=1)

	# Saving model architecture
	config = model.get_config()
	model = Sequential.from_config(config)
	json_string = model.to_json()
	model = model_from_json(json_string)
	with open("model.json", "w") as json_file:
		json_file.write(json_string)	
		
	model.summary()

	# Saving model weights
	model.save_weights('model.h5')

	predict=model.predict_classes(Test_X)
	print (predict)
	print (Test_Y)

	# compute the ConfusionMat
	ct=confusion_matrix(Test_Y,predict)
   
	# check the order of the CT
	order=np.unique(np.concatenate((predict,Test_Y)))
	
	#create an array to hold the CT for each CV
	mat=np.zeros((n_exp,n_exp))
	#put the order accordingly, in order to form the overall ConfusionMat
	for m in range(len(order)):
		for n in range(len(order)):
			mat[int(order[m]),int(order[n])]=ct[m,n]
		   
	tot_mat=mat+tot_mat
	   

	# write each CT of each CV into .txt file
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

		

def modify_cam(model, classes):
	model.pop()
	model.pop()		
	model.pop()
	model.pop()
	model.pop()
	model.pop()
	model.add(GlobalAveragePooling2D(data_format='channels_first'))
	model.add(Dense(classes, activation = 'softmax'))	
	return model


def VGG_16_cam(classes, channel_first=True, weights_path=None):


	model = Sequential()
	if channel_first:
		model.add(ZeroPadding2D((1,1), input_shape=(classes, 224, 224)))
	else:
		model.add(ZeroPadding2D((1,1), input_shape=(224, 224, classes)))


	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))


	model.add(GlobalAveragePooling2D(data_format='channels_first'))

	model.add(Dense(classes, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)


	return model


def modify_cam(model, classes):
	model.pop()
	model.pop()		
	model.pop()
	model.pop()
	model.pop()
	model.pop()
	model.add(GlobalAveragePooling2D(data_format='channels_first'))
	model.add(Dense(classes, activation = 'softmax'))	
	return model


	########### Image Data Generator ##############
	image_generator = ImageDataGenerator(
		zca_whitening = True,
		rotation_range = 0.2,
		width_shift_range = 0.2,
		height_shift_range = 0.2, 
		zoom_range = 0.2,
		horizontal_flip = True,
		rescale = 1.5)
	###############################################

		elif train_spatial_flag == 1 and train_temporal_flag == 0 and cam_visualizer_flag == 0:
			# trains spatial module ONLY, no escape
			
			image_generator.fit(X)
			vgg_model.fit_generator(image_generator.flow(X, y, batch_size=batch_size,
			 save_to_dir="./augmented/", save_format='png', save_prefix='augmented_me'),
			  steps_per_epoch=len(X)/batch_size, epochs=spatial_epochs)
			
			# Spatial Training
			if tensorboard_flag == 1:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
			else:
				vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)

			vgg_model.save_weights(spatial_weights_name + str(sub) + ".h5")
			plot_model(vgg_model, to_file="spatial_module_ONLY.png", show_shapes=True)




			# Testing
			# predict = vgg_model.predict_classes(test_X, batch_size = batch_size)
			# Test_Y_gt = np.repeat(Test_Y_gt, timesteps_TIM, axis=0)

			# For Majority Vote (make batch size divisible by 10(TIM No.))
			predict = vgg_model.predict_classes(test_X, batch_size = batch_size)
			voted_predict = []
			i = 0
			while i < int(len(predict)/timesteps_TIM) - 1:
				fraction_of_predict = predict[i * timesteps_TIM : (i+1) * timesteps_TIM]
				# print(fraction_of_predict)
				fraction_of_predict = np.asarray(fraction_of_predict)
				frequencies = np.bincount(fraction_of_predict)
				highest_frequency = np.argmax(frequencies)
				voted_predict += [highest_frequency]

				i += 1
				if i+1 >= int(len(predict)/timesteps_TIM) :
					fraction_of_predict = predict[(i) * timesteps_TIM : len(predict)]
					fraction_of_predict = np.asarray(fraction_of_predict)
					frequencies = np.bincount(fraction_of_predict)
					highest_frequency = np.argmax(frequencies)
					voted_predict += [highest_frequency]					

			# print(voted_predict)
			predict = voted_predict	

		elif train_spatial_flag == 0 and train_temporal_flag == 1:
			# trains temporal module ONLY.

			# Temporal Training
			if tensorboard_flag == 1:
				temporal_model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=temporal_epochs, callbacks=[tbCallBack])
			else:
				temporal_model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=temporal_epochs)	
				# temporal_model.train_on_batch(Train_X, Train_Y)
			temporal_model.save_weights(temporal_weights_name + str(sub) + ".h5")

			# Testing
			predict = temporal_model.predict_classes(Test_X, batch_size = batch_size)

		elif svm_flag == 1 and finetuning_flag == 0:
			# no finetuning

			X = vgg_model.predict(X, batch_size=batch_size)
			y_for_svm = np.argmax(y, axis=1)

			svm_classifier.fit(X, y_for_svm)

			test_X = vgg_model.predict(test_X, batch_size=batch_size)
			predict = svm_classifier.predict(test_X)

			Test_Y_gt = np.repeat(Test_Y_gt, timesteps_TIM, axis=0)


# all autoencoding
			model_ae = Model(inputs=conv_ae.input, outputs=conv_ae.output)
			plot_model(model_ae, to_file='autoencoders.png', show_shapes=True)

			# Autoencoding
			output_ae = model_ae.predict(normalized_X, batch_size = batch_size)

			for i in range(batch_size):
				visual_ae = output_ae[i].reshape(224,224,channel)
				# de-normalize
				visual_ae = ( ( visual_ae - min(visual_ae) ) / ( max(visual_ae) - min(visual_ae) ) ) * 255
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix='AE_output', index=str(sub),
				 												hash=np.random.randint(1e7), format='png')
				cv2.imwrite(db_home+'Classification/Result/ae_train/'+fname, visual_ae)
				
			output_ae = model.predict(output_ae, batch_size = batch_size)

			output_ae = model_ae.predict(normalized_test_X, batch_size = batch_size)
			for i in range(batch_size):
				visual_ae = output_ae[i].reshape(224,224,channel)
				# de-normalize
				visual_ae = ( ( visual_ae - min(visual_ae) ) / ( max(visual_ae) - min(visual_ae) ) ) * 255
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix='AE_output', index=str(sub),
				 												hash=np.random.randint(1e7), format='png')
				cv2.imwrite(db_home+'Classification/Result/ae_train/'+fname, visual_ae)

			output_ae = model.predict(output_ae, batch_size = batch_size)

# blurry
# foveated_image = image
# foveated_image = cv2.blur(foveated_image, (20, 20))
# foveated_image[from_x:to_x, from_y:to_y] = cropped_image