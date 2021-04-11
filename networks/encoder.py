import pdb
import torch
import torch.nn as nn
import torchvision
from .vgg import Encoder_vgg
# from torchsummary import summary
if __name__ == '__main__':
	from nnutils import conv_unit
else:
	from .nnutils import conv_unit

class Encoder(nn.Module):
	def __init__(self, pretrained_path, device, network, parameter_fix=False):
		super(Encoder, self).__init__()
		self.network = network
		# self.encoder_list = nn.ModuleList(list(Encoder_vgg(network=network, in_channels=3, pretrained_path=pretrained_path).features.to(device)))
		self.encoder_list = list(Encoder_vgg(network=network, in_channels=3, pretrained_path=pretrained_path).features.to(device))
		if parameter_fix == True:
			print('Parameter Fix!')
			for encoder in self.encoder_list:
				for param in encoder.parameters():
					# print(param.requires_grad)
					param.requires_grad = False
					# print(param.requires_grad)
					# import sys
					# sys.exit()
		# if network == 'uent':
		# self.conv1x1 = conv_unit(in_ch=512, out_ch=512, kernel_size=1, activation='relu').to(device)

	def forward(self, x):
		ft_list = []
		for model_i, model in enumerate(self.encoder_list):
			x = model(x)
			if model_i % 2 == 0:
				ft_list.append(x)
		# if self.network == 'uent':
		# x = self.conv1x1(x)
		return x, ft_list[:]