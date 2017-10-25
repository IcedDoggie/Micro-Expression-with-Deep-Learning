from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
from keras.optimizers import SGD

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import Callback
from keras import optimizers
from keras import metrics
from keras import backend as K
from keras.layers import GlobalAveragePooling2D

def global_average_pooling(x):
	return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
	return input_shape[0:2]

def VGG16_convolutions():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
	return model

def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3, 224, 224)))
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
	model.add(MaxPooling2D((2,2), strides=(2,2))) # 33

	model.add(Flatten())
	model.add(Dense(4096, activation='relu')) # 34
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu')) # 35
	model.add(Dropout(0.5))
	model.add(Dense(2622, activation='softmax')) # Dropped


	if weights_path:
		model.load_weights(weights_path)
	model.pop()
	model.add(Dense(5, activation='softmax')) # 36
	
	return model

def get_model():
	model = VGG16_convolutions()

	model = load_model_weights(model, "vgg16_weights.h5")
	
	model.add(Lambda(global_average_pooling, 
			  output_shape=global_average_pooling_shape))

	model.add(Dense(2, activation = 'softmax', init = 'uniform'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
	model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
	return model

def modify_model(weights_path):
	model = VGG_16(weights_path)
	model.pop()
	model.pop()
	model.pop()
	model.pop()
	model.pop()
	model.pop()	
	# model.add(Lambda(global_average_pooling,
	# 		  output_shape = global_average_pooling_shape))	
	model.add(GlobalAveragePooling2D(data_format='channels_first'))
	model.add(Dense(5, activation = 'softmax'))
	adam = optimizers.Adam(lr=0.00001)

	# sgd = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.5, nesterov = True)
	model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = [metrics.categorical_accuracy])
	return model


def load_model_weights(model, weights_path):
	print( 'Loading model.' )
	f = h5py.File(weights_path)
	for k in range(f.attrs['nb_layers']):
		if k >= len(model.layers):
			# we don't look at the last (fully-connected) layers in the savefile
			break
		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		model.layers[k].set_weights(weights)
		model.layers[k].trainable = False
	f.close()
	print( 'Model loaded.' )
	return model

def get_output_layer(model, layer_name):
	# get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	# print(layer_dict)
	layer = layer_dict[layer_name]
	return layer