# Python Libraries
import numpy as np
import cv2
from PIL import Image
import os

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn



def weights_init(m):
    classname = m.__class__.__name__	
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)	
        m.bias.data.fill_(0)
    # print(m)
    # print("weights")
class TextEncoding(nn.Module):
	def __init__(self, nt, batch_size, vocab):
		super(TextEncoding, self).__init__()
		self.nt = nt
		self.batch_size = batch_size
		self.vocab = vocab
		self.sparse_layer = nn.Embedding(self.vocab, self.nt)
		self.sparse_layer.cuda()

		# self.rnn_model = nn.LSTM(40, 256, self.batch_size).cuda()
		max_length = 1024
		hidden_size = 256
		n_layers = 2
		# self.embedding = nn.Embedding(self.batch_size)
		# self.linear_layer =
		# self.linear_layer = nn.Linear() 
		self.rnn = nn.LSTM(max_length, hidden_size, n_layers, batch_first=True)
		self.rnn.cuda()
		# self.rnn.cuda()
		# self.sparse_layer = nn.Embedding(self.batch_size * self.cap_length + 1, self.nt).cuda()
		# self.linear_layer = nn.Linear( (self.cap_length + 1)*self.nt, self.nt ).cuda()
		# self.leaky_relu = nn.LeakyReLU(0.1).cuda()
	
	def forward(self, original_text, cap_length, h0):
		# original_text = self.sparse_layer(original_text)
		# original_text = original_text.float()
		original_text = original_text.cuda()
		original_text = self.sparse_layer(original_text)
		# print(original_text)
		packed_captions = torch.nn.utils.rnn.pack_padded_sequence(original_text, cap_length, batch_first=True)
		
		# print(cap_length)
		# packed_captions, lengths_padded = torch.nn.utils.rnn.pad_packed_sequence(packed_captions, batch_first=True)
		# print('original text')
		# print(packed_captions)
		# packed_captions = packed_captions
		# packed_captions = packed_captions.float()
		output, h0 = self.rnn(packed_captions)
		output = output.data
		first_dim = len(output)
		noise_z = torch.rand(128, 256)
		noise_z = Variable(noise_z).cuda()

		output = torch.cat([output, noise_z], 0)
		
		

		return output, h0
# text encoding
def text_encoding(vocab_len, original_text, nt, nz,cap_length, batch_size):
	# initialization of pytorch functions

	# Word embedding here
	text_embedding = nn.Embedding(batch_size * cap_length + 1, nt).cuda()

	# RNN layers
	# rnn = nn.LSTM(cap_length + 1, 256, batch_size).cuda()	

	# FC layer
	linear_layer = nn.Linear((cap_length + 1) * nt, nt).cuda()
	leaky_relu = nn.LeakyReLU(0.1).cuda()

	# pass to fully connected layer
	original_text = Variable(original_text).cuda()
	original_text = text_embedding(original_text)
	# print(original_text)
	# original_text = original_text.float()
	# print(original_text)
	original_text = original_text.view(batch_size, nt, cap_length + 1)
	
	# print(original_text)
	# original_text = original_text.view(args.batch_size, )	
	# print(original_text)
	# out, hidden_state = rnn(original_text)
	# print(out)
	# original_text = original_text.view(batch_size, (cap_length + 1) * nt )
	# print(original_text)
	output = linear_layer(original_text)
	output = leaky_relu(output) 
	# print(output)
	# noise concatenation
	dim_a = len(output)
	dim_b = len(output[0][:])
	dim_c = len(output[:][0][0])

	noise_z = torch.rand(dim_a, nz, dim_c)

	noise_z = Variable(noise_z).cuda()
	# print(noise_z)
	output = torch.cat([output, noise_z], 1)
	# print(output)
	return output


# def image_encoding(image, batch_size)	

# TODO: Text Decoder for sanity check
# def text_decoding(vocab, encoded_text):
# 	print("Decode for sanity-check")
# 	rnn_1 = nn.LSTM()



# 	return caption


# generator model
class Generator(nn.Module):
	def __init__(self, nz, nt, ngf, batch_size):
		# n_z -> no. of dim for noise
		# n_t -> no. of dim for text features
		# ngf -> no. of dim for gen filters in first conv layer
		super(Generator, self).__init__()
		self.batch_size = batch_size
		self.nt = nt
		self.nz = nz
		self.model = nn.Sequential(
			#TODO: Change conv to transpconv
			# print(nz+nt),
			# weights_init(Generator),
			nn.ConvTranspose2d(4608, ngf * 8, 4), # 356, 1024
			nn.BatchNorm2d(ngf * 8), # 1024

			# Feature learning # 1
			# for big images
			nn.Conv2d(ngf * 8, ngf * 2, 1, 1, 0), # 1024, 256
			nn.BatchNorm2d(ngf * 2), # 256
			nn.ReLU(True),
			# # nn.ReLU(True),
			nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1), # 256, 256																																																																				
			nn.BatchNorm2d(ngf * 2), # 256
			nn.ReLU(True),
			nn.Conv2d(ngf * 2, ngf * 8, 3, 1, 1), # 256, 1024
			nn.BatchNorm2d(ngf * 8), # 1024
			nn.ReLU(True),
			# nn.LeakyReLU(0.1, True),

			# # TODO: conv again for bigger image

			# #### start to turn into image~~~~
			# # for big images
			# # state size : (ngf * 8) * 4 * 4 
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1), # 1024, 512
			nn.BatchNorm2d(ngf * 4), # 512
			# nn.LeakyReLU(0.1, True),

			# Feature learning # 2
			# # # state size: (ngf * 4) * 8 * 8
			nn.Conv2d(ngf * 4, ngf * 1, 1, 1, 0), # 512, 128
			nn.BatchNorm2d(ngf * 1), # 128
			nn.ReLU(True),
			nn.Conv2d(ngf * 1, ngf * 1, 3, 1, 1), # 128, 128
			nn.BatchNorm2d(ngf * 1), # 128
			nn.ReLU(True),
			nn.Conv2d(ngf * 1, ngf * 4, 3, 1, 1), # 128, 512
			nn.BatchNorm2d(ngf * 4), # 512 
			nn.ReLU(True),

			# # # TODO: conv again for bigger image

			# # state size: (ngf * 4) * 8 * 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
			# # nn.Conv2d(ngf * 4, ngf * 2, 1, 1), # 512, 256 
			nn.BatchNorm2d(ngf * 2), # 256
			nn.LeakyReLU(0.1, True),

			# Feature learning # 3
			# nn.Conv2d(ngf * 2, ngf * 1, 1, 1, 0), # 512, 128
			# nn.BatchNorm2d(ngf * 1), # 128
			# nn.ReLU(True),
			# nn.Conv2d(ngf * 1, ngf * 1, 3, 1, 1), # 128, 128
			# nn.BatchNorm2d(ngf * 1), # 128
			# nn.ReLU(True),
			# nn.Conv2d(ngf * 1, ngf * 2, 3, 1, 1), # 128, 512
			# nn.BatchNorm2d(ngf * 2), # 512 
			# nn.ReLU(True),

			# # state size: (ngf * 2) * 16 * 16
			nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1), # 256, 128
			nn.BatchNorm2d(ngf * 1), # 128
			nn.LeakyReLU(0.1, True),



			# # # # state size: (ngf) * 32 * 32, 2nd param is no. of channel(hardcoded)
			# nn.ConvTranspose2d(ngf * 1, 3, 1, 1, 1), # 128, 3
			nn.ConvTranspose2d(ngf * 1, ngf * 1, 4, 2, 1),
			nn.BatchNorm2d(ngf),
			nn.LeakyReLU(0.1, True),

			# nn.ConvTranspose2d(ngf * 1, 3, 4, 2, 1),
			nn.ConvTranspose2d(ngf * 1, 3, 1),
			# # # # state size: (num_of_channel) * 64 * 64  (Imageeee~~~baby~~)
			# weights_init(Generator),

			# # # # activation
			nn.Tanh(),
			# weights_init(Generator)

			
			)


	def forward(self, x):

		# a_c_dim has to match with the first conv layer

		x = x.view(self.batch_size, 4608, 1, 1).cuda()
		# print("Launch Generator!")	

		# print(x)
		out = self.model(x)
		# print(out)
		out = out.view(self.batch_size, 3, 64, 64).cuda()
		# out = out.view(16, 3, 64, 64).cuda()

		return out

# discriminator model
class Discriminator(nn.Module):
	def __init__(self, nz, nt, ngf, ndf, batch_size):
		self.nt = nt
		self.batch_size = batch_size
		# n_z -> no. of dim for noise
		# n_t -> no. of dim for text features
		# ngf -> no. of dim for gen filters in first conv layer
		# ndf -> no. of dim for discriminator filters in first conv layer
		super(Discriminator, self).__init__()
		# weights_init(Discriminator)
		self.model = nn.Sequential(
			# nn.Conv2d(3, ndf * 1, 4, 2, 1),
			nn.Conv2d(3, ndf * 1, 1),
			nn.LeakyReLU(0.2, True),

			nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, True),

			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
			nn.BatchNorm2d(ndf * 4),
			# nn.LeakyReLU(0.1, True),
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
			nn.BatchNorm2d(ndf * 8),

			
			# nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
			# nn.BatchNorm2d(ndf * 8),

			# bigger image possibly
			nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, True),

			nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, True),

			nn.Conv2d(ndf * 2, ndf * 8, 6),
			# nn.Conv2d(ndf * 2, ndf * 8, 2),
			# nn.Conv2d(ndf * 2, ndf * 8, 3, 3, 1),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, True),

			)


		self.model2 = nn.Sequential(
			nn.Conv2d(ndf * 8 + nt, ndf * 8, 1),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, True),

			# nn.Conv2d(ndf * 8, ndf * 4, 1),
			# nn.BatchNorm2d(ndf * 4),
			# nn.LeakyReLU(0.2, True),

			nn.Conv2d(ndf * 8, 1, 3),
			# nn.BatchNorm2d(1),
			# nn.LeakyReLU(0.2, True),

			# nn.Conv2d(ndf * 1, 1, 4),
			nn.Sigmoid()

			)


	def forward(self, x, text, cap_length):
		
		# a_c_dim has to match with the first conv layer
		x = x.view(self.batch_size, 3, 64, 64)

		###### model 1 #######
		# text that describes generated image
		out = self.model(x)
		# print(out)
		# out = out.view(4, 8192, 1, 1)
		# print(out)	
		# text input layers
		linear_layer2 = nn.Linear(cap_length + 1, self.nt)
		linear_layer2.cuda()
		# batch_norm2 = nn.BatchNorm2d(self.nt)
		leaky2 = nn.LeakyReLU(0.2,  True)

		####### text input ##########
		# print(text)
		# print(cap_length)

		# text_v = Variable(text)
		text_v = text
		text_v = text_v.float().cuda()
		text_input = linear_layer2(text_v)	
		# text_input = batch_norm2()
		text_input = leaky2(text_input)
		# print(text_input)
		# print("text input")
		# print(len(text_input[0]))
		reshape_len = len(text_input[1])   # hardcoded
		# print(reshape_len)
		text_input = text_input.view(self.batch_size, 1024, 1, 1).cuda() # hardcoded batchsize

		padded_text = nn.functional.pad(text_input, 
			(1, 1, 1, 1), 'replicate').cuda()
		

		####### concat text and output from model1 #########
		# print(out)
		# print(padded_text)

		_cat = torch.cat((padded_text, out), 1)
		# print(_cat)

		####### TODO: pass whole concatenated structure to model2 ########
		out = self.model2(_cat)


		return out