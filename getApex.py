import pandas as pd
import os, shutil


filepath = "/media/ice/OS/Datasets/CASME2_Cropped/"
excelpath = "/media/ice/OS/Datasets/CASME2_label_Ver_2.xls"
target_dest = "/media/ice/OS/Datasets/CASME2_Apex/"


apex = pd.read_excel(excelpath)
apex = apex[["Subject", "Filename", "ApexFrame", "Estimated Emotion"]]

############ create folders for each subject ############
counter = 1
while counter < 27:
	if counter < 10:
		dest = target_dest + "sub0" + str(counter)
	else:
		dest = target_dest + "sub" + str(counter)
	os.mkdir(dest)
	counter += 1
#########################################################

text_filename = "expression_labels.txt"
text_path = "/media/ice/OS/Datasets/" + text_filename
file = open(text_path, 'a')


####################### Find Apex Frame ###########################
counter = 0
while counter < len(apex):
	idx = apex.loc[counter]
	subject = idx[['Subject']].Subject
	filename = idx[['Filename']].Filename
	apexframe = idx[['ApexFrame']].ApexFrame
	expression = idx[['Estimated Emotion']].values[0]
	# print(expression)

	# format frame number -> jpg
	if type(apexframe) == int:
		if (apexframe) < 10:
			frameNo = "00" + str(apexframe) + ".jpg"
		elif (apexframe) < 100:
			frameNo = "0" + str(apexframe) + ".jpg"
		else:
			frameNo = str(apexframe) + ".jpg"

		if subject < 10:
			subject = "sub0" + str(subject)
		else:
			subject = "sub" + str(subject)
		# Final prepending path
		source = filepath + subject + "/" + str(filename) + "/" + frameNo
		target = target_dest + subject + "/" + str(counter) + ".jpg"
		
		try:
			shutil.copyfile(source, target)
		except:
			print(source + "File not exists! @.@")
		print(target)

	###### get labels ######
	if expression == 'happiness':
		result = 0
	elif expression == 'disgust':
		result = 1
	elif expression == "repression":
		result = 2
	elif expression == "surprise":
		result = 3
	elif expression == "others":
		result = 4
	file.write(str(result) + '\n')
	########################

	counter += 1

######################################################################

