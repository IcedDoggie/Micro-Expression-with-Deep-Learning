import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pydot, graphviz


from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.applications.vgg16 import VGG16 as keras_vgg16
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import keras
from keras.callbacks import EarlyStopping


from utilities import LossHistory
from models import VGG_16, temporal_module



root = "/media/ice/OS/Datasets/"
emotion_path = "/media/ice/OS/Datasets/CK_TIM10/Emotion_labels/"
image_path = "/media/ice/OS/Datasets/CK_TIM10/CK_TIM10/"
dataset = "CK_TIM10/"

def read_emotion_labels(emotion_path):
	emotion_vec = []
	for root, item, txt in os.walk(emotion_path):
		if len(root) > 51:
			txt_path = root + '/' + txt[0]
			txt_np = np.loadtxt(txt_path)
			txt_np = str(int(txt_np))
			emotion_vec += [txt_np]
	emotion_np = np.asarray(emotion_vec)



	return emotion_np

def read_image(image_path):
	
	image_vec = []
	for root, item, img_list in os.walk(image_path):
		if len(root) > 46:
			# counter = 0
			
			for img in img_list:
				img_path = root + '/' + img
				image = cv2.imread(img_path)
				image = cv2.resize(image, (224, 224))
				# image = image.flatten()

				image_vec += [image]

				# if counter == 9:
				# 	tim_image_vec += [image_vec]

				# counter += 1
	
	image_vec = np.asarray(image_vec)
	image_vec = image_vec.reshape((image_vec.shape[0], image_vec.shape[3], image_vec.shape[1], image_vec.shape[2]))
	# tim_image_vec = np.reshape(())
	# print(image_vec.shape)


	return image_vec


def binarize_emotion_labels(emotion_np, classes):

	emotion_np = emotion_np.astype(np.uint8)
	emotion_np = np.subtract(emotion_np, 1)
	y = np_utils.to_categorical(emotion_np, classes)

	return y
def train():

	root = '/home/ice/Documents/Micro-Expression/'

	# training config
	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min')
	batch_size = 1
	spatial_epochs = 100
	temporal_epochs = 100

	# load data
	emotion_np = read_emotion_labels(emotion_path)
	image_vec = read_image(image_path)

	# preprocess emotion_np
	y = binarize_emotion_labels(emotion_np, int(max(emotion_np)))
	y = np.repeat(y, 10, axis=0)
	print(y.shape)

	# optimizer
	adam = optimizers.Adam(lr=0.00001, decay=0.0000001)

	# weights_name
	spatial_weights_name = root + "ck_spatial_weights.h5"
	temporal_weights_name = root + "ck_temporal_weights.h5"
	vgg_weights = root + "VGG_Face_Deep_16.h5"

	# models
	# print(int(max(emotion_np)))
	vgg_model = VGG_16(224, int(max(emotion_np)), 3, weights_path=vgg_weights)
	lstm_model = temporal_module(4096, 10, int(max(emotion_np)))
	vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
	lstm_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])


	# train
	for layer in vgg_model.layers[:33]:
		layer.trainable = False

	print(image_vec.shape)
	vgg_model.fit(image_vec, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[history, stopping])
	vgg_spatial_encoder = Model(inputs=vgg_model.input, outputs=model.layers[35].output)
	# plot_model(model, to_file = "ck_training.png", show_shapes=True)	

	spatial_features = vgg_spatial_encoder.predict(image_vec, batch_size=batch_size)
	lstm_model.fit(spatial_features, y, batch_size=batch_size, epochs=temporal_epochs)

	vgg_spatial_encoder.save_weights(spatial_weights_name)
	lstm_model.save_weights(temporal_weights_name)


	
# train()
# tim_image_vec = read_image(image_path)

