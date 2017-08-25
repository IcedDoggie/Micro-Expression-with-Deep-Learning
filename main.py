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

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr



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
	VidPerSubject = [9,13,7,5,19,5,9,3,13,13,10,12,8,4,3,4,34,3,15,11,2,2,12,7,7,16]
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



##### Setting up the LSTM model ########
data_dim=r*w # 2500
print(data_dim)
timesteps=10

# LSTM1 = LSTM(2500, return_sequences=True, input_shape=(timesteps, data_dim))

model=Sequential()
# model.add(TimeDistributed(Dense(data_dim), input_shape=(timesteps, data_dim)))
model.add(LSTM(2500, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(500,return_sequences=False))
##model.add(LSTM(500,return_sequences=True))
##model.add(LSTM(50,return_sequences=False))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(5,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=[metrics.categorical_accuracy])
#### generate the label based on subjects #########
label=np.loadtxt(workplace+'Classification/'+ dB +'_label.txt')
labelperSub=[]
counter = 0
for sub in range(subjects):
	numVid=VidPerSubject[sub]
	labelperSub.append(label[counter:counter+numVid])
	counter = counter + numVid
##print(np.shape(labelperSub[1]))
##print(labelperSub[1])

######## Seperating the input files into LOSO CV ########
tot_mat=np.zeros((n_exp,n_exp))
for sub in range(subjects):
	Train_X=[]
	Train_Y=[]

	Test_X=SubperdB[sub]
	Test_X=np.array(Test_X)
	Test_Y=labelperSub[sub]
	Test_Yy=np_utils.to_categorical(Test_Y,5)
	print(Test_Y)
##    print(np.shape(Test_Y))
	if sub==0:
		for i in range(1,subjects):
			Train_X.append(SubperdB[i])
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
				
	# print(Train_X)
	# Train_X=np.hstack(Train_X)
	# print(Train_X.shape)
	Train_X=np.vstack(Train_X) # changed to hstack from vstack
	# print(Train_X.shape)
	# Train_X = Train_X.shape[1:]
	# print(Train_X.shape)
	# Train_X = np.expand_dims(Train_X, axis=2)
	# Train_X = np.reshape(Train_X, Train_X.shape + (1, 1,) )
	# Train_X = np.reshape( Train_X, Train_X.shape )
	# Train_X = np.reshape(2500, 16077)	
	print(Train_X.shape)

	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y,5)
	print (np.shape(Train_Y))
	print (np.shape(Train_X))
	print (np.shape(Test_Y))	
	print (np.shape(Test_X))
	model.fit(Train_X, Train_Y, validation_split=0.05, epochs=1, batch_size=20)
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
##    predict[predict>= 0.5] = 1
##    predict[predict<0.5] = 0
	print (predict)
	print (Test_Y)

	#compute the ConfusionMat
	ct=confusion_matrix(Test_Y,predict)
   
	#check the order of the CT
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

		

