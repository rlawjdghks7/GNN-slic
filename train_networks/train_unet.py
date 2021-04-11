import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import numpy as np
from tqdm import tqdm
import shutil
import copy

from utils.utils import *


def train_unet(encoder ,decoder, dataloader, optimizer, criterion, nas_dir, device, i, epochs=10):
	encoder_model_wts = copy.deepcopy(encoder.state_dict())
	decoder_model_wts = copy.deepcopy(decoder.state_dict())
	best_loss = 1e10

	nas_dir = make_dir(nas_dir, 'iter_{}'.format(i))
	save_dir = make_dir(nas_dir, 'val_result', remove=True)

	for epoch in range(epochs):
		print('Epoch {} / {}'.format(epoch, epochs-1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				encoder.train()
				decoder.train()
			else:
				encoder.eval()
				decoder.eval()

			epoch_loss = 0.0
			cnt = 0
			total_dsc = 0.0

			for image, label, _, slic_arr, superpix_img, _, img_path in tqdm(dataloader[phase]):
				slic_arr = slic_arr[0]
				img_path = img_path[0]
				img_name = os.path.basename(img_path)
				image = image.to(device)
				label = label.to(device)
				# print(image.size())
				# sys.exit()
				original_size = image.size()[2:]
				optimizer.zero_grad()
				

				with torch.set_grad_enabled(phase=='train'):
					x_encode, ft_list = encoder(image.float())
					output, _ = decoder(x_encode, ft_list)
					# output = decoder(x_encode, ft_list)

					loss = criterion(output, label)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				epoch_loss += loss.item()
				cnt += 1
				if phase == 'val' and epoch == epochs-1:
					dsc = save_unet_img(image, label, output, slic_arr, superpix_img, save_dir, img_name)
					total_dsc += dsc

			print('[{}]loss : {:.4f}'.format(phase, epoch_loss/cnt))
			if phase == 'val' and epoch == epochs-1:
				result_txt = os.path.join(nas_dir, 'val_result.txt')
				f = open(result_txt, 'w')
				f.write('dsc: {:.4f}'.format(total_dsc))
				print('total dsc : {:.4f}'.format(total_dsc/cnt))
			if phase == 'val' and epoch_loss/cnt < best_loss:
				print('saving best model')
				best_loss = epoch_loss/cnt
				encoder_model_wts = copy.deepcopy(encoder.state_dict())
				decoder_model_wts = copy.deepcopy(decoder.state_dict())

	print('best val loss: {:.4f}'.format(best_loss))
	encoder.load_state_dict(encoder_model_wts)
	decoder.load_state_dict(decoder_model_wts)
	
	return encoder, decoder