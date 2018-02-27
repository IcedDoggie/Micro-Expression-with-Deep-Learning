import argparse
from train import train
from train_samm import train_samm
from train_samm_cross import train_samm_cross
from test_samm_cross import test_samm_cross
from train_cae_lstm import train_cae_lstm
from test_casme import test_casme
def main(args):
	# print(args[0]['train'])
	if args.train == "./train.py":
		train(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	# train_smic(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id)
	elif args.train == "./train_samm.py":
		train_samm(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	elif args.train == "./train_samm_cross.py":
		train_samm_cross(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	elif args.train == "./test_samm_cross.py":
		test_samm_cross(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	elif args.train == "./train_cae_lstm.py":
		train_cae_lstm(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)
	elif args.train == "./test_casme.py":
		test_casme(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.tensorboard)



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