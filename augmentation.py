from imgaug import augmenters as iaa
import imgaug as ia

from reordering import readinput

import cv2
import numpy as np
import os
import imutils

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

def cube_maker(image):
	image = cv2.resize(image, (280, 280))
	return image

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

for sub in sorted([infile for infile in os.listdir(img_path)]):     

		for vid in sorted([inrfile for inrfile in os.listdir(img_path+sub)]):
			
			path=img_path + sub + '/'+ vid + '/'
			if path in listOfIgnoredSamples:
				continue
			print(path)
			imgList=readinput(path,dB)
		  
			numFrame=len(imgList)

			for var in range(numFrame):
				img=cv2.imread(imgList[var])
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

				augmentations = [ rotated_img, rotated_img_2, rotated_img_3, rotated_img_4, 
				scaled_img0_9, scaled_img1_0, scaled_img1_1, translation_1, translation_2, translation_3,
				translation_4, horizontal, vertical ]


				counter = 0
				# print(augmentations)
				while counter < 13:
					filename = path + "0" + str(numFrame + 1 + counter) + ".jpg"
					cv2.imwrite(filename, augmentations[counter])
					counter += 1
					new_image += 1

				print("Saved to " + path)

print("Total new images: " + str(new_image))			


