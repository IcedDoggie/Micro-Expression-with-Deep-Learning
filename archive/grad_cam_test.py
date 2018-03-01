import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import pydot, graphviz
from PIL import Image

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
from vis.visualization import visualize_cam, overlay, visualize_activation
from vis.backprop_modifiers import guided
from vis.grad_modifiers import relu

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, loading_smic_table, loading_casme_table, ignore_casme_samples, ignore_casmergb_samples, LossHistory
from utilities import loading_samm_table
from models import VGG_16, temporal_module, modify_cam, VGG_16_4_channels, convolutional_autoencoder


# model = 'VGG_Face_Deep_16.h5'

vgg_model_cam = VGG_16(spatial_size=224, classes=5, weights_path='vgg_spatial_50_CASME2_Optical_0.h5')
vgg_model = Model(inputs=vgg_model_cam.input, outputs=vgg_model_cam.output)
plot_model(vgg_model, to_file="keras-vis.png", show_shapes=True)

vgg_model_cam = Model(inputs=vgg_model_cam.input, outputs=vgg_model_cam.layers[29].output)
plot_model(vgg_model_cam, to_file="keras-vis-cam.png", show_shapes=True)

image = '002.jpg'
image = cv2.imread(image)
image = cv2.resize(image,(224,224))
output = visualize_cam(vgg_model_cam, 29, 0, image)
output2 = visualize_cam(vgg_model_cam, 29, 1, image)
output3 = visualize_cam(vgg_model_cam, 29, 2, image)
output4 = visualize_cam(vgg_model_cam, 29, 3, image)
output5 = visualize_cam(vgg_model_cam, 29, 4, image)
# guide = guided(vgg_model)

activation = visualize_activation(vgg_model, 29, None, image)
# output6 = visualize_cam(vgg_model_cam, 29, 5, image)
# output7 = visualize_cam(vgg_model_cam, 29, 6, image)
overlaying = overlay(image, output)
overlaying2 = overlay(image, output2)
overlaying3 = overlay(image, output3)
overlaying4 = overlay(image, output4)
overlaying5 = overlay(image, output5)
overlay_act = overlay(image, activation)


print(output.shape)
# output = output.reshape(224, 224, 3)
# overlaying = overlaying.reshape(224, 224, 3)
# cv2.imwrite('ccam.png', output)
cv2.imwrite('coverlayingcam.png', overlaying)
cv2.imwrite('coverlayingcam2.png', overlaying2)
cv2.imwrite('coverlayingcam3.png', overlaying3)
cv2.imwrite('coverlayingcam4.png', overlaying4)
cv2.imwrite('coverlayingcam5.png', overlaying5)
cv2.imwrite('overlayact.png', overlay_act)

image_vgg = image.reshape(1, 3, 224, 224)
prediction = vgg_model.predict(image_vgg)
print(prediction)

# cv2.imwrite('ccam2.png', output2)
# cv2.imwrite('ccam3.png', output3)
# cv2.imwrite('ccam4.png', output4)
# cv2.imwrite('ccam5.png', output5)
# cv2.imwrite('ccam6.png', output6)
# cv2.imwrite('ccam7.png', output7)