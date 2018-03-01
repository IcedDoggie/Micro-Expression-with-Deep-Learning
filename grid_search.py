import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

plt.figure()
title = "test"
classes = ['2nd Last FC', 'Last FC', '', '', ''] # VGG
classes_2 = ['1-layer LSTM(512)', '1-layer LSTM(3000)', '2-layer LSTM(1024-1024)', '2-layer LSTM(3000-1024)', '2-layer LSTM(5000-1024)'] # LSTM
cmap = plt.cm.Blues
sample_list = np.array([[0, 0.35], [0, 0.35], [0, 0.31], [0.27, 0.31], [0, 0.27]])

plt.imshow(sample_list, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes_2)

thresh = sample_list.max() / 2.
for i, j in itertools.product(range(sample_list.shape[0]), range(sample_list.shape[1])):
	plt.text(j, i, format(sample_list[i, j]),
				 horizontalalignment="center",
				 color="white" if sample_list[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()