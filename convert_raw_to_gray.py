import cv2
import os
import shutil

path_to_raw = "/media/ice/OS/Datasets/CASME2-RAW_images/CASME2-RAW/"


for sub, vid, files in os.walk(path_to_raw):
	# if 'videos' in sub or 'in_range' in sub:
		# print(sub)
	if len(sub) > 57:
		for file in files:
			path = sub + '/' + file
			img = cv2.imread(path)
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			cv2.imwrite(path, gray)
			print(path)
	# 	if 'in_range' in sub or 'videos' in sub or 'video' in sub:
	# 		shutil.rmtree(sub)
		# shutil.rmtree(sub)
		# if "in_range" not in sub:
			# print(files)
	# print(sub)