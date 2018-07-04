from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential, Model
from keras import metrics


from reordering import readinput


import cv2
import numpy as np
import os
import imutils
from matplotlib import pyplot as plt

def rotation(degree, image):
	# M = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), degree, 1 )
	# images_aug = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
	images_aug = imutils.rotate(image, degree)
	images_aug = cv2.cvtColor(images_aug, cv2.COLOR_BGR2GRAY)


	return images_aug

def scaling(ratio, image):

	res = cv2.resize(image,None,fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
	return res

def translation(move_x, move_y, image):
	# M = np.float32([[ 1, 0, move_x],[ 0, 1, move_y]])
	# dst = cv2.warpAffine( image, M, (image.shape[0], image.shape[1]) )
	dst = imutils.translate(image, move_x, move_y)
	return dst

def flip(image):
	horizontal = cv2.flip(image, 1)
	vertical = cv2.flip(image, 0)

	return horizontal, vertical

def pixel_removal(number_of_pixels, image_file):

	image = cv2.imread(image_file)
	# list_of_images = np.empty([0])
	list_of_images = []

	# preliminaries
	size_y, size_x = image.shape[0], image.shape[1]
	removal_region = 9
	diff_x = int(size_x/2) - number_of_pixels
	diff_y = int(size_y/2) - number_of_pixels
	d_x = 0
	d_y = 0



	for i in range(removal_region):
		image_removal = cv2.imread(image_file, 1)


		if d_x > size_x:
			d_x = 0

		# 3, 4, 5
		if i == 3:
			d_y += diff_y

		# 6, 7, 8
		if i == 6:
			d_y += diff_y			


		image_removal[d_y:d_y+number_of_pixels, d_x:d_x+number_of_pixels] = 0

		d_x += diff_x


		# list_of_images = np.append(list_of_images, image_removal)
		list_of_images += [image_removal]
	return list_of_images




img_path = "/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/"
export_path = "./"
dB="CASME2_TIM"


IgnoredSamples=['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
				'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
				'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/']
listOfIgnoredSamples=[]
for s in range(len(IgnoredSamples)):
	if s==0:
		listOfIgnoredSamples=[img_path+IgnoredSamples[s]]
	else:
		listOfIgnoredSamples.append(img_path+IgnoredSamples[s])
resizedFlag = 1
r=50; w=50

new_image = 0
def augmenting():
	new_image = 0
	for sub in sorted([infile for infile in os.listdir(img_path)]):     

			for vid in sorted([inrfile for inrfile in os.listdir(img_path+sub)]):
				augmentations = np.empty([0])
				path = img_path + sub + '/'+ vid + '/'
				folder_path = img_path + sub + '/'
				if path in listOfIgnoredSamples:
					continue
				# print(path)
				imgList=readinput(path,dB)
			  	
				numFrame=len(imgList)
				# print(numFrame)

				# hardcoded
				number_of_ops = 13
				folder_num = 0
				
				for var in range(numFrame):

					img = cv2.imread(imgList[var])
					rotated_img = rotation(5, img)
					rotated_img_2 = rotation(-5, img)
					rotated_img_3 = rotation(10, img)
					rotated_img_4 = rotation(-10, img)				
					scaled_img0_9 = scaling(0.9, img)
					scaled_img1_0 = scaling(1.0, img)
					scaled_img1_1 = scaling(1.1, img)
					translation_1 = translation(-2, -2, img)
					translation_2 = translation(-2, 2, img)
					translation_3 = translation(2, -2, img)
					translation_4 = translation(2, 2, img)
					horizontal, vertical = flip(img)

					augmentations = np.append(augmentations, [ rotated_img, rotated_img_2, rotated_img_3, rotated_img_4, 
					scaled_img0_9, scaled_img1_0, scaled_img1_1, translation_1, translation_2, translation_3,
					translation_4, horizontal, vertical ])




				counter = 0
				seq_var = -9
				count_seq = 0
				filename_count = 1

				while counter < len(augmentations):
					
					# split augmented data into different folders
					folder_name = vid + "_" + str(folder_num)
					folder_name = folder_path + folder_name + "/"
					# print(folder_name)
					if not os.path.exists(folder_name):
						os.makedirs(folder_name)	

					# filename = path + "0" + str( numFrame + counter + 1 ) + ".jpg"
					# filename = folder_name + "0" + str( numFrame + counter + 1 ) + ".jpg"
					if filename_count < 10:
						filename = folder_name + "00" + str( filename_count ) + ".jpg"
					else:
						filename = folder_name + "0" + str( filename_count ) + ".jpg"

					# print(str(seq_var))
					print(filename)
					os.chdir(folder_name)
					cv2.imwrite(filename, augmentations[seq_var + 13])
					os.chdir(path)
					counter += 1
					new_image += 1
					seq_var += 13
					filename_count += 1
					if counter % 10 == 0 and counter > 0:
						count_seq += 1
						seq_var = -13 + count_seq 
						folder_num += 1
						filename_count = 1
						
					# print("Saved to " + path)
				# seq_var += 1

	print("Total new images: " + str(new_image))			


def cube_maker():
	counter = 1
	for sub in sorted([infile for infile in os.listdir(img_path)]):     

			for vid in sorted([inrfile for inrfile in os.listdir(img_path+sub)]):
				
				path=img_path + sub + '/'+ vid + '/'
				if path in listOfIgnoredSamples:
					continue
				print(path)
				imgList=readinput(path,dB)
			  	
				numFrame=len(imgList)

				for var in range(numFrame):

					img = cv2.imread(imgList[var])
					img = cv2.resize(img, (256, 256))
					cv2.imwrite(imgList[var], img)
					counter += 1

					print(str(counter) + "/5659" )

# cube_maker()
def preprocessing_keras():
	
	datagen = ImageDataGenerator(
		horizontal_flip = True,
		)


	train_generator = datagen.flow_from_directory(
		directory = '/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/', 
		target_size = (280, 280),
		batch_size = 26,
		class_mode = None,
		save_to_dir = '/media/ice/OS/Datasets/CASME2_TIM/augment/',
		save_prefix = '',
		save_format = 'jpeg',
		shuffle = 'false'
		)
	# print(train_generator.filenames)

	for X_item in train_generator:
		print("augmented")


def augmenting2():
	new_image = 0
	for sub in sorted([infile for infile in os.listdir(img_path)]):     

		for vid in sorted([inrfile for inrfile in os.listdir(img_path+sub)]):
			augmentations = np.empty([0])
			path = img_path + sub + '/'+ vid + '/'
			folder_path = img_path + sub + '/'
			if path in listOfIgnoredSamples:
				continue
			# print(path)
			imgList=readinput(path,dB)
		  	
			numFrame=len(imgList)
			# print(numFrame)
			# print("imglist: %i" % imgList)

			for var in range(numFrame):
				augmentations = pixel_removal(2, imgList[var])
				
				# print("augmentations: %i" % len(augmentations))
				# print("var: %i" % var)
				counter = 0


				while counter < len(augmentations):
						
				# 	# split augmented data into different folders
					folder_name = vid + "_" + str(counter)
					folder_name = folder_path + folder_name + "/"
					# print(folder_name)
					if not os.path.exists(folder_name):
						os.makedirs(folder_name)	
					if var < 9:
						filename = folder_name + "00" + str( var + 1 ) + ".jpg"
					else:
						filename = folder_name + "0" + str( var + 1 ) + ".jpg"

					os.chdir(folder_name)
					cv2.imwrite(filename, augmentations[counter])
					os.chdir(path)
					counter += 1
					print("Written to: " + filename)
				
					new_image += 1
	print("Total new images: " + str(new_image))			



augmenting2()