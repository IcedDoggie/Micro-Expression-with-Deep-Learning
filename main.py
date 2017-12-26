import argparse
from train import train

def main(args):
	train(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	# train_smic(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id)


	# flag list:
	# st -> spatio-temporal
	# s -> spatial only
	# t -> temporal only
	# nofine - > no finetuning, train svm classifer only
	# scratch -> train from scratch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default='./train.py', help='Using which script to train.')
	parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
	parser.add_argument('--spatial_epochs', type=int, default=10, help='Epochs to train for Spatial Encoder')
	parser.add_argument('--temporal_epochs', type= int, default=40, help='Epochs to train for Temporal Encoder')
	parser.add_argument('--train_id', type=str, default="0", help='To name the weights of model')
	parser.add_argument('--dB', type=str, default='CASME2_TIM', help='Specify Database')
	parser.add_argument('--spatial_size', type=int, default=224, help='Size of image')
	parser.add_argument('--flag', type=str, default='st', help='Flags to control type of training')
	parser.add_argument('--tensorboard', type=bool, default=False, help='tensorboard display')

	args = parser.parse_args()
	print(args)

	main(args)