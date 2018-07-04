# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import string
from faceplusplus import face_landmarking
from PIL import Image
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight
 
		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth
	def align(self, image, gray, rect):
		shape = self.predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye (x, y)-coordinates
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]    	

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)
 
		# return the aligned face
		# return output  
		return output, M, w, h              

def landmark_extraction(shape_predictor, image, rects, rect_flag):

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)
	fa = FaceAligner(predictor, desiredFaceWidth=256)
	
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(image)
	img_w, img_h, c = image.shape
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
	# detect faces in the grayscale image
	if rect_flag == 0:
		rects = detector(gray, 1)


	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	 
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)

		# print('x:{0} y:{1} w:{2} h:{3}'.format(x, y, w, h))
		cropped_image = image[y:y+h, x:x+w]
		
		aligned_image, M, w, h = fa.align(image, gray, rect)
		# aligned_image = aligned_image[30:210, 50:210]
		# aligned_image = cv2.resize(aligned_image, (280,340))

	return aligned_image, M, w, h



output_path = "/home/ice/Documents/Micro-Expression/aligned_SAMM/"
images_path = "/home/ice/Documents/Micro-Expression/SAMM/"

# create directories in cropped_SAMM
def create_dir_in_target_folder():
	helper_flag = 0
	counter = 0
	for subject, video, files in os.walk(images_path):
		if len(video) > 0 and helper_flag == 0:
			subject_list = video
			for item in subject_list:
				file_output = output_path + item
				os.mkdir(file_output)
			helper_flag = 1


		elif len(video) > 0:
			for item in video:
				file_output = output_path + subject_list[counter] + '/' + item
				os.mkdir(file_output)

			counter += 1

# create_dir_in_target_folder()

# main logic run
counter = 1
for subject, video, files in os.walk(images_path):

	if len(subject) > 45:

		rects = 0
		rect_flag = 0
		for item in files:
			filepath = subject + '/' + item
			file_output = filepath.replace('SAMM', 'aligned_SAMM')

			if rect_flag == 0:
				# w, h, left, top = face_landmarking(filepath)
				test, M, width, height = landmark_extraction(args['shape_predictor'], filepath, rects, rect_flag)
				cv2.imwrite('helper.jpg', test)
				helper = 'helper.jpg'
				w, h, left, top = face_landmarking(helper)
				rect_flag = 1

			# cv2.imwrite('helper.jpg', aligned_image)
			
			image = cv2.imread(filepath)
			ori_h, ori_w = image.shape[0], image.shape[1]
			image = cv2.resize(image, (500,338))
			image = cv2.warpAffine(image, M, (height, width), flags=cv2.INTER_CUBIC)
			# image = cv2.resize(image, (ori_w, ori_h))

			cropped_image = image[top:top+h, left:left+w]			
			cropped_image = cv2.resize(cropped_image, (280,340))

			# cv2.imshow('cropped', cropped_image)
			# cv2.waitKey(0)

			rect_flag = 1
			# cv2.imshow('asd', cropped_image)
			# cv2.waitKey(0)
			cv2.imwrite(file_output, cropped_image)

		print("%i/159"%counter)
		counter += 1

	 
# landmark_extraction(args['shape_predictor'], args['image'])