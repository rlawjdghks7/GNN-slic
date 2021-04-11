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

def test_unet(encoder ,decoder, dataloader, criterion, nas_dir, device, i):
	encoder.eval()
	decoder.eval()

	nas_dir = make_dir(nas_dir, 'iter_{}'.format(i))
	save_dir = make_dir(nas_dir, 'result', remove=True)

	with torch.no_grad():
		total_loss = 0.0
		cnt = 0
		total_dsc = 0.0
		for image, label, adj, slic_arr, supix_img, graph_label, adj_label, img_path in tqdm(dataloader):
			img_name = os.path.basename(img_path[0])
			slic_arr = slic_arr[0]
			image = image.to(device)
			label = label.to(device)
			original_size = image.size()[2:]

			x_encode, ft_list = encoder(image.float())
			output, dft_list = decoder(x_encode, ft_list)
			# output = torch.sigmoid(output)

			loss = criterion(output, label)
			total_loss += loss.item()
			dsc = save_unet_img(image, label, output, slic_arr, superpix_img, save_dir, img_name)
			total_dsc += dsc
			if cnt < 1:
				save_original_img(image, label, superpix_img, nas_dir, img_name)
				for i, feature_map in enumerate(ft_list):
					ft_interpolated = F.interpolate(feature_map, original_size)
					show_featuremaps(ft_interpolated, slic_arr, nas_dir, img_name, 'encoder_{}'.format(i+1))
				for i, feature_map in enumerate(dft_list):
					ft_interpolated = F.interpolate(feature_map, original_size)
					show_featuremaps(ft_interpolated, slic_arr, nas_dir, img_name, 'decoder_{}'.format(i+1))
				
			cnt += 1
			# sys.exit()
		result_txt = os.path.join(nas_dir, 'result.txt')
		f = open(result_txt, 'w')
		f.write('dsc: {:.4f}'.format(total_dsc))

		print('total dsc : {:.4f}'.format(total_dsc/cnt))
		print('total loss : {:.4f}'.format(total_loss/cnt))

	return total_dsc, total_loss