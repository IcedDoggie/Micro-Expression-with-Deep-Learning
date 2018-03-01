# Python Libraries
import argparse
import numpy as np
import cv2
import pickle
import os
import time

# PyTorch Libraries
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.utils.data


# Local Libraries
from dataloader import MsDataset, get_loader
from build_vocab import Vocabulary
from models import Generator, Discriminator, text_encoding, weights_init, TextEncoding

# Tensorboard
from tensorboard_logger import configure, log_value

def save_checkpoint(state, filename = 'checkpoint.pth'):
	torch.save(state, filename)
	# if is_best:
	# 	shutil.copyfile(filename, 'model_best.pth')

def main(args):
	########## for tensorboard namings #############
	file_counter = 1000
	filename = "runs/run-1000"
	while os.path.isdir(filename):
		file_counter += 1
		filename = "runs/run-" + str(file_counter)
	print(filename)
	configure(filename)
	################################################

	########## Load Text Encoder #############

	##########################################

	if(args.cuda==True):
		cudnn.benchmark = True
		torch.backends.cudnn.enabled = True
	else:
		cudnn.benchmark = False
		torch.backends.cudnn.enabled = False


	# Image preprocessing
	transform = transforms.Compose([ 
		transforms.RandomCrop(args.crop_size),
		transforms.RandomHorizontalFlip(), 
		transforms.ToTensor(), 
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# Load Vocabulary
	file_vocab = open(args.vocab_path, 'rb')
	vocab = pickle.load(file_vocab)
	words = vocab.word2idx	
	print(words)

	# Build data loader
	data_loader, length_data = get_loader(args.root_dir, args.image_dir, args.caption_path, 
							args.vocab_path, transform, args.batch_size)

	# Load pretrained model if exists, or create if it isn't
	G = Generator(args.nz, args.nt, args.ngf, args.batch_size)
	G.apply(weights_init)	
	G.cuda()

	if args.netG:
		G.load_state_dict(torch.load(args.netG))
		# print(G)
		print("loaded model")

	D = Discriminator(args.nz, args.nt, args.ngf, args.ndf, args.batch_size)
	D.apply(weights_init)
	D.cuda()

	if args.netD:		
		D.load_state_dict(torch.load(args.netD))
		# print(D)
		print("loaded D")

	# Create FC-RNN-FC layer
	# text_encoder = TextEncoding(args.nt, args.batch_size)

	# Loss and Optimizer
	criterion = nn.BCELoss(size_average=True)

	# RNN
	rnn = TextEncoding(args.nt, args.batch_size, len(vocab))
	max_length = 40
	hidden_size = 40
	n_layers = 30
	# rnn = nn.LSTM(max_length, hidden_size, n_layers, batch_first=True)
	# rnn.cuda()

	# Training credentials
	real_labels = Variable(torch.ones(1)).cuda()
	fake_labels = Variable(torch.zeros(1)).cuda()
	k_t = 0
	h0 = Variable(torch.randn(n_layers, args.batch_size, hidden_size))
	x0 = Variable(torch.randn(args.batch_size, 1, max_length))
	fixed_lengths = []
	fixed_lengths.extend(range(1,65))

	for index, item in enumerate(fixed_lengths):
		fixed_lengths[index] = max_length

	for epoch in range(args.num_epochs):
		
		if epoch % 10 == 0:
			learning_rate = args.learning_rate * args.lr_decay
		else:
			learning_rate = args.learning_rate

		d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
		g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

		for i, (images, captions, lengths) in enumerate(data_loader):

			############ Data Preparation ############
			### images : load image
			### text   : encoded_text
			# TODO: need to make even size captions
			
			# Compute time
			start = time.time()
			
			real_labels = Variable(torch.ones(args.batch_size)).cuda()
			fake_labels = Variable(torch.zeros(args.batch_size)).cuda()
			captions_length = len(captions[1].numpy()) - 1
			images = Variable(images).cuda()
			captions = Variable(captions).cuda()
			
			encoded_text, h0 = rnn.forward(captions, lengths, h0)

			##########################################

			############# Train Discriminator ###############
			D.zero_grad()
			
			# Train with Real
			d_output_real = D.forward(images, captions, captions_length)
			d_loss_real = criterion(d_output_real, real_labels)
			d_loss_real.backward(retain_variables=True)

			################ Don't use this ####################
			# Train with Fake
			# encoded_text = encoded_text.data
			encoded_text_firstdim = len(encoded_text)
			encoded_text = encoded_text.view(256, encoded_text_firstdim)
			# print(encoded_text)			
			fc_layer = nn.Linear(encoded_text_firstdim , args.nt + args.nz)
			fc_layer.cuda()
			encoded_text = fc_layer(encoded_text)
			# print(encoded_text)

			fake = G.forward(encoded_text)
			output = D.forward(fake, captions, captions_length)
			d_loss_fake = criterion(output, fake_labels)
			d_loss_fake.backward(retain_variables=True)

			# Train with Wrong
			wrong = D.forward(images, captions, captions_length)
			d_loss_wrong = criterion(wrong, fake_labels)
			d_loss_wrong.backward(retain_variables=True)

			# Getting D_LOSS
			d_loss = d_loss_real + d_loss_wrong + d_loss_fake / 2
			d_loss.backward(retain_variables=True)
			d_optimizer.step()
			###################################################
			
			############## Train Generator ###################
			G.zero_grad()

			g_output = G.forward(encoded_text)

			# discriminate fake images
			d_output_fake = D.forward(g_output, captions, captions_length)
			g_loss = criterion(d_output_fake, real_labels)
			
			# Getting G_LOSS
			g_loss.backward(retain_variables=True)
			g_optimizer.step()

			# d_loss =  d_loss_real + g_loss

			# d_optimizer.step()
			##################################################
			
			# Decode input text for sanity check


			# Reshape output based on batch size
			g_output = g_output.view(args.batch_size, 3, 64, 64)
			
			end = time.time()
			diff = end - start			
			# print(g_output)
			if True:
				print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
					'D(x): %.2f, D(G(z)): %.2f, Computation_time: %.5f' 
					%(epoch, 200, (i + 1)*args.batch_size, length_data, d_loss.data[0], g_loss.data[0],
					d_output_real.cpu().data.mean(), d_output_fake.cpu().data.mean(), diff))			
			
			# torchvision.utils.save_image(g_output.data,
			#  './output/generated_sample_%d.png' % (epoch+1))
			torchvision.utils.save_image(g_output.data,
			 args.output_path + '/generated_sample_%d.png' % (epoch+1))
			# visualization
			log_value('g_loss', g_loss.data[0], i)
			log_value('d_loss', d_loss.data[0], i)

	

			# Memory free-ups
			# del g_output, d_output_fake, d_output_real, images, d_loss_real, d_loss_fake, g_loss, d_loss

			if i >= 203:
				break
		if epoch % 1 == 0:
			epoch_count = 0
			torch.save(G.state_dict(), args.checkpoint_path + '/netG_epoch_%d.pth' % (epoch_count))
			torch.save(D.state_dict(), args.checkpoint_path + '/netD_epoch_%d.pth' % (epoch_count))
		# epoch += 1
		continue
		print(epoch)

# call main and do add_argument
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/' ,
						help='path for saving trained models')
	parser.add_argument('--root_dir', type=str, default='./', 
						help='path of the root directory.')
	parser.add_argument('--crop_size', type=int, default=64 ,
						help='size for randomly cropping images')
	parser.add_argument('--vocab_path', type=str, default='./Data/vocab_coco.pickle',
						help='path for vocabulary wrapper')
	parser.add_argument('--image_dir', type=str, default='./resized_coco/' ,
						help='directory for resized images')
	parser.add_argument('--caption_path', type=str,
						default='./',
						help='path for train annotation json file')
	# parser.add_argument('--log_step', type=int , default=10,
	# 					help='step size for prining log info')
	# parser.add_argument('--save_step', type=int , default=1000,
	# 					help='step size for saving trained models')
	parser.add_argument('--cuda', action='store_true',
						help='use cuda?')	
	parser.add_argument('--output_path', type=str, default='./output', 
						help='specify output path please :)')
	parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', 
						help='specify checkpoint path please :)')
	parser.add_argument('--netG', type=str, default='', help='load netG pretrained')
	parser.add_argument('--netD', type=str, default='', help='load netD pretrained')

	
	# Model parameters
	# parser.add_argument('--embed_size', type=int , default=256 ,
	# 					help='dimension of word embedding vectors')
	# parser.add_argument('--hidden_size', type=int , default=512 ,
	# 					help='dimension of lstm hidden states')
	# parser.add_argument('--num_layers', type=int , default=1 ,
	# 					help='number of layers in lstm')

	# Model parameters
	parser.add_argument('--ngf', type=int, default=128, 
		help='dim of gen filters for first layer')
	parser.add_argument('--nz', type=int, default=128, 
		help='dim of noise')
	parser.add_argument('--nt', type=int, default=1024, 
		help='dim of text features')	
	parser.add_argument('--ndf', type=int, default=64, 
		help='dim of text features')		

	# TODO: add ndf for discriminator
	
	parser.add_argument('--num_epochs', type=int, default=5000)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--learning_rate', type=float, default=0.0005)
	parser.add_argument('--lr_decay', type=float, default=0.5)
	parser.add_argument('--lambda_k', type=float, default=1)
	parser.add_argument('--gamma', type=float, default=1)
	args = parser.parse_args()
	print(args)

	main(args)
