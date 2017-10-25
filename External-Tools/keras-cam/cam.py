from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
from data import *
import cv2
import argparse

import pydot, graphviz
from keras.utils import np_utils, plot_model

def visualize_class_activation_map(model_path, img_path, output_path, run_count):

	model = modify_model(model_path)
	original_img = cv2.imread(img_path, 1)
	original_img = cv2.resize(original_img, (224, 224))
	width, height, _ = original_img.shape

	# Reshape to the network input shape (3, w, h).
	img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

	# Get the 512 input weights to the softmax.
	model = Model(inputs=model.input, outputs=model.layers[32].output)
	plot_model(model, to_file="vgg_original.png", show_shapes=True)
	class_weights = model.layers[-1].get_weights()[0]
	layer_target = "conv2d_" + str(run_count)
	final_conv_layer = get_output_layer(model, layer_target)
	get_output = K.function([model.layers[0].input],
	[final_conv_layer.output, model.layers[-1].output])

	[conv_outputs, predictions] = get_output([img])
	predicted_class = np.argmax(predictions)

	if predicted_class == 0:
		predicted_class = "happiness"
	elif predicted_class == 1:
		predicted_class = "disgust"
	elif predicted_class == 2:
		predicted_class = "repression"
	elif predicted_class == 3:
		predicted_class = "surprise"
	else:
		predicted_class = "others"

	conv_outputs = conv_outputs[0, :, :, :]

	# print(predictions)
	# Create the class activation map.
	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
	for i, w in enumerate(class_weights[:, 1]):
		cam += w * conv_outputs[i, :, :]

	cam /= np.max(cam)
	cam = cv2.resize(cam, (height, width))
	heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
	heatmap[np.where(cam < 0.2)] = 0
	img = heatmap*0.5 + original_img
	
	output_path = output_path + "_" + predicted_class + ".jpg"
	cv2.imwrite(output_path, img)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
	parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
	parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
	parser.add_argument("--model_path", type = str, help = "Path of the trained model")
	parser.add_argument("--dataset_path", type = str, help = \
	'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
	http://pascal.inrialpes.fr/data/human/')
	args = parser.parse_args()
	return args

# if __name__ == '__main__':
	# args = get_args()

	# if args.train:
	# train(args.dataset_path)
	# else:
	# visualize_class_activation_map(args.model_path, args.image_path, args.output_path)
	
model_path = 'vgg_spatial_ID_12.h5'
img_path_array = []
img_path = '/media/ice/OS/Datasets/CASME2_TIM/CASME2_TIM/'
out_path = '/home/ice/Documents/Micro-Expression/External-Tools/keras-cam/CASME2_output/'
first_run = 1

for root, dirnames, filenames in os.walk(img_path):
	if len(dirnames) > 0:
		if first_run == 1:
			first_run = 0
			subject_path_array = dirnames
		else:
			img_path_array += [dirnames]
	files = filenames

counter = 0
final_path = np.empty([0])
output_path = np.empty([0])
IgnoredSamples = ['sub09/EP13_02/','sub09/EP02_02f/','sub10/EP13_01/','sub17/EP15_01/',
'sub17/EP15_03/','sub19/EP19_04/','sub24/EP10_03/','sub24/EP07_01/',
'sub24/EP07_04f/','sub24/EP02_07/','sub26/EP15_01/']
ignore_flag = 0
for subject in subject_path_array:
	path_array = img_path_array[counter]
	for item in path_array:
		for file in files:
			
			path_to_parse = img_path + str(subject) + '/' + str(item) + '/' + str(file)
			file = file[0:3]
			out_parse = out_path + str(subject) + '/' + str(item) + '/' + str(file)
			for ignorance in IgnoredSamples:
				if ignorance in path_to_parse:
					ignore_flag = 1

				if ignore_flag == 0:
					final_path = np.append(final_path, path_to_parse)
					output_path = np.append(output_path, out_parse)
				else:
					ignore_flag = 0



	counter += 1


heatmap_count = 0
run_count = 13
for item in final_path:
	heatmap_path = output_path[heatmap_count]
	visualize_class_activation_map('./vgg_spatial_ID_12.h5', item, heatmap_path, run_count)
	run_count += 13
	heatmap_count += 1
	print(str(heatmap_count) + "/2460 processed" + "\n")
