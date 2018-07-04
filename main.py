import argparse
from train import train
from train_samm_cross import train_samm_cross
from test_samm_cross import test_samm_cross
from train_cae_lstm import train_cae_lstm
# from test_casme import test_casme
from train_spatial_only import train_spatial_only
from train_ram import train_ram

def main(args):
	# print(args[0]['train'])
	print(args.objective_flag)
	if args.train == "./train.py":
		train(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	# train_smic(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id)
	elif args.train == "./train_samm_cross.py":
		train_samm_cross(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	elif args.train == "./test_samm_cross.py":
		test_samm_cross(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	elif args.train == "./train_cae_lstm.py":
		train_cae_lstm(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	# elif args.train == "./test_casme.py":
	# 	test_casme(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	elif args.train == "./train_spatial_only.py":
		train_spatial_only(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)
	elif args.train == "./train_ram.py":
		train_ram(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)




	# flag list:
	# st -> spatio-temporal
	# s -> spatial only
	# t -> temporal only
	# nofine - > no finetuning, train svm classifer only
	# scratch -> train from scratch

	# eg for calling more than 1 databases:
	# python main.py --dB 'CASME2_Optical' 'CASME2_Strain_TIM10' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st4se'

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default='./train.py', help='Using which script to train.')
	parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
	parser.add_argument('--spatial_epochs', type=int, default=10, help='Epochs to train for Spatial Encoder')
	parser.add_argument('--temporal_epochs', type= int, default=40, help='Epochs to train for Temporal Encoder')
	parser.add_argument('--train_id', type=str, default="0", help='To name the weights of model')
	parser.add_argument('--dB', nargs="+", type=str, default='CASME2_TIM', help='Specify Database')
	parser.add_argument('--spatial_size', type=int, default=224, help='Size of image')
	parser.add_argument('--flag', type=str, default='st', help='Flags to control type of training')
	parser.add_argument('--objective_flag', type=int, default=1, help='Flags to use either objective class or emotion class')
	parser.add_argument('--tensorboard', type=bool, default=False, help='tensorboard display')


	args = parser.parse_args()
	print(args)

	main(args)