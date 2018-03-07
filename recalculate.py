from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np



def cal_f1(gt, pred):
	cm = confusion_matrix(gt, pred)
	diagonal = cm.diagonal()
	print(cm)
	f1 = []
	for i in range(len(cm[0])):
		col_sum = sum(cm[:, i])
		row_sum = sum(cm[i])
		precision = diagonal[i] / col_sum
		recall = diagonal[i] / row_sum
		f1_score = 2 * (precision * recall) / (precision + recall + 0.00001)
		f1 += [f1_score]
	print(f1) 
	# print(diagonal)
	f1 = sum(f1) / 5
	return f1
file_path = '/home/ice/Documents/Micro-Expression/Results/log_cde3_composite_corrected.txt'
table = pd.read_table(file_path, names=['vid_name', 'ground_truth', 'prediction'], sep=' ')
table = table.dropna()
table = table.drop(['vid_name'], axis=1)
table = table.astype(int)
table = table.as_matrix()
gt = table[:, 0]
pred = table[:, 1]


f1 = cal_f1(gt, pred)
print(f1)