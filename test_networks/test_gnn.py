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

import matplotlib.pyplot as plt

from train_networks.dice import dice_score
from utils.utils import *
from utils.build_graph import gen_graph, gen_adj
from utils.gen_graph_label import gen_graph_label

def reconstruct_graph(output, slic_arr, idx):
	predict_img = np.zeros((slic_arr.shape))
	
	for i, val in enumerate(output):
		mask = (slic_arr == i)*(val[idx].data.cpu())
		# if val.data.cpu() > 0.05:
		# 	mask = (slic_arr == i)*1
		# else:
		# 	mask = (slic_arr == i)*0
		predict_img += mask.numpy()

	return predict_img
		
def save_graphfeatruemap(graph, slic_arr, nas_dir, img_name, graph_name):
	for_testing = make_dir(nas_dir, 'test_feature')
	img_test_dir = make_dir(for_testing, img_name.replace('.jpg', ''))
	graph_name_dir = make_dir(img_test_dir, graph_name)

	for i in range(graph.size()[1]):
		graph_feature_path = os.path.join(graph_name_dir, '{}_graph.png'.format(i))
		graph_img = normalize_output(reconstruct_graph(graph, slic_arr, i))
		plt.imsave(graph_feature_path, graph_img, cmap='gray')


def test_gnn(encoder, gnn, dataloader, criterion, nas_dir, device, i, concat=False):
	encoder.eval()
	gnn.eval()

	total_dsc = 0.0
	total_loss = 0.0
	cnt = 0

	# nas_dir = make_dir(nas_dir, 'iter_{}'.format(i), remove=True)
	nas_dir = make_dir(nas_dir, 'iter_{}'.format(i))
	save_dir = make_dir(nas_dir, 'result', remove=False)

	with torch.no_grad():
		for image, label, adj, slic_arr, superpix_img, graph_label, adj_label, img_path in tqdm(dataloader):
			img_path = img_path[0]
			img_name = os.path.basename(img_path)
			# print(img_name)
			slic_arr = slic_arr[0]

			image = image.to(device)
			label = label.to(device)
			adj = adj[0].to(device)
			
			graph_label = graph_label[0].to(device).double()

			original_size = image.size()[2:]
			x_encode, ft_list = encoder(image.float())
			# print(x_encode.size())
			# sys.exit()
			
			x_interpolated = F.interpolate(x_encode, original_size)
			if concat == True:
				ft_to_graph_list = []
				for i, ft_map in enumerate(ft_list[:-1]):
					ft_interpolated = F.interpolate(ft_map, original_size)
					gft = gen_graph(ft_interpolated, slic_arr).to(device)
					ft_to_graph_list.append(gft)

			graph = gen_graph(x_interpolated, slic_arr).to(device)
			if concat == True:
				output, gft_list = gnn(graph.float(), adj, ft_to_graph_list)
			else:
				output, gft_list = gnn(graph.float(), adj)
			
			# graph_label = gen_graph_label(label, slic_arr)
			# graph_label = graph_label.to(device).double()

			loss = criterion(output.double(), graph_label)
			dsc = save_predict_img(image, label, output, slic_arr, superpix_img, save_dir, img_name)
			
			
			total_dsc += dsc
			total_loss += loss
			# if cnt < 1:
			save_original_img(image, label, superpix_img, nas_dir, img_name)
			for i, feature_map in enumerate(ft_list):
				ft_interpolated = F.interpolate(feature_map, original_size)
				show_featuremaps(ft_interpolated, slic_arr, nas_dir, img_name, 'encoder_{}'.format(i+1))
			for i, gft in enumerate(gft_list):
				save_graphfeatruemap(gft, slic_arr, nas_dir, img_name, 'graph_{}'.format(i+1))
			save_graphfeatruemap(graph, slic_arr, nas_dir, img_name, 'graph_{}'.format(0))
			# if cnt == 3:
			
			# sys.exit()
			cnt += 1

		result_txt = os.path.join(nas_dir, 'result.txt')
		f = open(result_txt, 'w')
		f.write('dsc: {:.4f}'.format(total_dsc))

	print('total dsc : {:.4f}'.format(total_dsc/cnt))
	print('total loss : {:.4f}'.format(total_loss/cnt))

	return total_dsc, total_loss