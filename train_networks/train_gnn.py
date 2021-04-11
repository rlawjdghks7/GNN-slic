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
from utils.build_graph import gen_graph
from utils.gen_graph_label import gen_graph_label


def reconstruct_graph(output, slic_arr, idx):
	predict_img = np.zeros((slic_arr.shape))
	
	for i, val in enumerate(output):
		mask = (slic_arr == i)*(val[idx].data.cpu())
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

def train_gnn (encoder, gnn, dataloader, optimizer, criterion, nas_dir, device, i, epochs=10, concat=False):
	encoder_model_wts = copy.deepcopy(encoder.state_dict())
	gnn_model_wts = copy.deepcopy(gnn.state_dict())
	best_loss = 1e10

	nas_dir = make_dir(nas_dir, 'iter_{}'.format(i))
	save_dir = make_dir(nas_dir, 'val_result', remove=True)
	
	for epoch in range(epochs):
		print('Epoch {} / {}'.format(epoch, epochs-1))
		print('-' * 10)
		
		for phase in ['train', 'val']:
			if phase == 'train':
				encoder.train()
				gnn.train()
			else:
				encoder.eval()
				gnn.eval()

			epoch_loss = 0.0
			cnt = 0
			total_dsc = 0.0
			# for image, label, adj, slic_arr, supix_img, graph_label, adj_label, img_path in tqdm(dataloader[phase]):
			for image, label, adj, slic_arr, supix_img, graph_label, img_path in tqdm(dataloader[phase]):
				img_name = os.path.basename(img_path[0])
				img_path = img_path[0]
				slic_arr = slic_arr[0]
				
				image = image.to(device)
				label = label.to(device)
				adj = adj[0].to(device)
				graph_label = graph_label[0].to(device).double()

				original_size = image.size()[2:]
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase=='train'): # val no backpropa
					x_encode, ft_list = encoder(image.float())
					x_interpolated = F.interpolate(x_encode, original_size)
					
					if concat == True:
						ft_to_graph_list = []
						for i, ft_map in enumerate(ft_list[:-1]):
							ft_interpolated = F.interpolate(ft_map, original_size)
							gft = gen_graph(ft_interpolated, slic_arr).to(device)
							ft_to_graph_list.append(gft)
					
					# sys.exit()
					graph = gen_graph(x_interpolated, slic_arr).to(device)
					if concat == True:
						output, _ = gnn(graph.float(), adj, ft_to_graph_list)
					else:
						output, _ = gnn(graph.float(), adj)
					print(output.size())
					print(graph_label.size())
					sys.exit()
					loss = criterion(output.double(), graph_label)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				epoch_loss += loss.item()
				cnt += 1
				
				if phase == 'val' and epoch == epochs-1:
					dsc = save_predict_img(image, label, output, slic_arr, supix_img, save_dir, img_name)
					total_dsc += dsc
			# np.set_printoptions(threshold=sys.maxsize)
			# print(attention.cpu().detach().numpy())
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
				gnn_model_wts = copy.deepcopy(gnn.state_dict())
	
	print('best val loss: {:.4f}'.format(best_loss))
	encoder.load_state_dict(encoder_model_wts)
	gnn.load_state_dict(gnn_model_wts)

	return encoder, gnn