import argparse
from train import train_casme2

def main(args):
	train_casme2(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default='./train.py', help='Using which script to train.')
	parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
	parser.add_argument('--spatial_epochs', type=int, default=40, help='Epochs to train for Spatial Encoder')
	parser.add_argument('--temporal_epochs', type= int, default=40, help='Epochs to train for Temporal Encoder')
	parser.add_argument('--train_id', type=int, default=0, help='To name the weights of model')

	args = parser.parse_args()
	print(args)

	main(args)