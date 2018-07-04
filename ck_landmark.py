import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

data_path = '/media/ice/OS/Datasets/ck/extended-cohn-kanade-images/'
cropped_path = '/media/ice/OS/Datasets/ck/cropped_ck/'
annotation_path = '/media/ice/OS/Datasets/ck/Emotion_labels/'

def create_dir(data_path):
	for root, file, images in os.walk(data_path):
		if len(root) > 54 and len(root) <= 58:
			root = root.replace('extended-cohn-kanade-images', 'cropped_ck')
			# print(root)
			os.mkdir(root)

	for root, file, images in os.walk(data_path):
		if len(root) > 58:
			root = root.replace('extended-cohn-kanade-images', 'cropped_ck')
			os.mkdir(root)


def read_data(data_path):
	# getting the data path
	folders = []
	for root, file, images in os.walk(data_path):
		if len(root) > 58:
			for image in images:
				file_path = root + "/" + image
				folders += [file_path]
	return folders

def crop_data(data_path):
	# create directory in cropped folder
	folders = read_data(data_path)
	# create_dir(data_path)

	for item in folders:

		# landmarks
		landmark_data = item.replace('.png', '_landmarks.txt')
		landmark_data = landmark_data.replace('extended-cohn-kanade-images', 'Landmarks')			
		landmarks = pd.read_table(landmark_data, header=None, sep="   ", names=['X', 'Y'])
		landmarks = landmarks.as_matrix()
		landmarks = landmarks.astype(int)

		left_most = min(landmarks[:, 0])
		right_most = max(landmarks[:, 0])
		up_most = max(landmarks[:, 1])
		down_most = min(landmarks[:, 1])

		image = cv2.imread(item)
		image = image[down_most-20:up_most, left_most:right_most]

		# saving/visualizing cropped image
		target_path = item.replace('extended-cohn-kanade-images', 'cropped_ck')
		cv2.imwrite(target_path, image)
		print(target_path)
		# implt = plt.imshow(image)
		# plt.show()

def annotation_cross_check(cropped_path, annotation_path, remove_empty_annotation, remove_missing_labels):

	existing_annotation = []
	missing_annotation = []
	no_annotation = []
	existing_vid = []

	for root, file, data in os.walk(annotation_path):
		if len(root) > 46:

			if len(data) > 0:
				data = data[0]
				data = data.replace("_emotion.txt", "")
			elif len(data) == 0:
				root = root.replace("Emotion_labels", "cropped_ck")
				missing_annotation += [root]

	for root, file, data in os.walk(cropped_path):
		if len(root) > 41:

			for item in data:
				item = item.replace('.png', '')
				
			for item in missing_annotation:
				if item == root:
					shutil.rmtree(item)
	

	# remove folders that are without annotations
	if remove_empty_annotation == True:
		for item in missing_annotation:
			item = item.replace("cropped_ck", "Emotion_labels")
			shutil.rmtree(item)

	for root, file, data in os.walk(annotation_path):
		if len(root) > 46:
			root = root.replace("Emotion_labels", "cropped_ck")
			existing_annotation += [root]

	for root, file, data in os.walk(cropped_path):
		if len(root) > 41:
			existing_vid += [root]

	missing = list(set(existing_vid) - set(existing_annotation))
	
	if remove_missing_labels == True:
		for item in missing:
			shutil.rmtree(item)





annotation_cross_check(cropped_path, annotation_path, False, True)