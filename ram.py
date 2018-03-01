import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os
import glob

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
import keras

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr

# Todo:
# 1. Glimpse Sensor
#    (a) glipmse network
#    (b) location network
# 2. Internal State, the hidden state of RNN and get updated by core network -> h_t = f_h(h_t-1, g_t ; theta_h) 
#    (a) First input: g_t from glimpse network
#    (b) Second input: hidden state in previous recurrent unit
#    (c) output: actions
# 3. Actions
#    (a) Input: Internal State
#    (b) First output: l_t, location to deploy its sensor
#    (c) Second output: classification, conditioned on both hidden state then theta_a. find out what is theta_a
# 4. Reward
#    (a) reward signal, r_t + 1
#    (b) accumulate and attempt to maximize the r, each correct classification moves reward closer to 1.
#        - 



class GlimpseNet(object):
	def __init__(self, images, loc):
		self.original_image_size = images.shape[0]
		self.glimpse_window_size = 16
		self.loc = loc

	def glimpse_sensor(self):
		image_subset = self.images
		
	def __call__(self):
		