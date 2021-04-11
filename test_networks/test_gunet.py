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

def prepare_attentional_mechanism_input(Wh):
		N = Wh.size()[0]
		
		Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
		Wh_repeated_alternating = Wh.repeat(N, 1)

		norm_result = torch.norm(Wh_repeated_alternating-Wh_repeated_in_chunks, dim=1)
		e = torch.exp((-1) * norm_result)
		# norm_result = torch.mean(torch.pow(Wh_repeated_alternating - Wh_repeated_in_chunks, 2), dim=1)
		# e = torch.exp((-1) * norm_result)
		result = e.view(N, N)
		norm_result = norm_result.view(N, N)

		# all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
		# result = all_combinations_matrix.view(N, N, 2 * self.out_features)

		return result, norm_result

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

	mean_feature = torch.mean(graph, dim=1).unsqueeze(1)	
	# mean_img = reconstruct_graph(mean_feature, slic_arr, 0)
	# plt.imshow(mean_img, cmap='gray')
	# plt.show()
	# sys.exit()

	for i in range(graph.size()[1]):
		graph_feature_path = os.path.join(graph_name_dir, '{}_graph.png'.format(i))
		graph_img = normalize_output(reconstruct_graph(graph, slic_arr, i))
		# graph_img = reconstruct_graph(graph, slic_arr, i)
		plt.imsave(graph_feature_path, graph_img, cmap='gray')
		# plt.imshow(graph_img, cmap='gray')
		# plt.show()


def test_gunet(encoder, gnn, dataloader, criterion, nas_dir, device, i, concat=False, network='gat'):
	encoder.eval()
	gnn.eval()

	total_dsc = 0.0
	total_loss = 0.0
	cnt = 0

	# nas_dir = make_dir(nas_dir, 'iter_{}'.format(i), remove=True)
	nas_dir = make_dir(nas_dir, 'iter_{}'.format(i))
	save_dir = make_dir(nas_dir, 'result', remove=True)

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
			adj_label = adj_label[0].to(device).double()

			original_size = image.size()[2:]
			x_encode, ft_list = encoder(image.float())
			# print(x_encode.size())
			# sys.exit()

			# # test norm
			# a = torch.from_numpy(np.array([[1, 2, 3, 4, 5], [6,7,8,9,10]])).float()
			# b = torch.from_numpy(np.array([[0, 1, 2, 3, 4], [5,6,7,8,9]])).float()
			# # a_2 = torch.pow(a-b, 2)
			# # print(a_2)
			# # norm = torch.mean(a_2, dim=1)
			# # print(norm)
			# norm = torch.norm(a-b, dim=1)
			# print(norm)
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
				output, gft_list, att_list, attention = gnn(graph.float(), adj, ft_to_graph_list)
			else:
				# output, gft_list, att_list, attention = gnn(graph.float(), adj)
				output, gft_down_list, gft_up_list, attention, score_list, recon_adj_list = gnn(graph.float(), adj)
			# for att in att_list:
			# 	plt.imshow(superpix_img[0])
			# 	# plt.imshow(slic_arr.cpu().numpy())
			# 	show_graph_with_labels(att.cpu().numpy(), slic_arr.numpy(), 87)
			# if 'Abyssinian_47' in img_name:
			# 	print(img_name)
				
			# 	# plt.imshow(superpix_img[0])
			# 	# # plt.imshow(slic_arr.cpu().numpy())
			# 	# # # N-250:110, 87, 69, 31, 27 / N-500:174, 186 / N-50:14, 31
			# 	# show_graph_with_labels(attention.cpu().numpy(), slic_arr.numpy(), 35)

			# 	for i in range(gft_up_list[-1].size()[1]):
			# 		feature = gft_up_list[-1][:,i].unsqueeze(1)
			# 		feature_img = normalize_output(reconstruct_graph(feature, slic_arr, 0))
			# 		# plt.imshow(feature_img, cmap='gray')
			# 		# show_graph_with_labels(attention.cpu().numpy(), slic_arr.numpy(), 31)
			# 		e, e_2 = prepare_attentional_mechanism_input(feature)
			# 		zero_vec = torch.zeros_like(e)
			# 		e_adj = torch.where(adj > 0, e, zero_vec)
			# 		sum_result = torch.sum(e_adj, dim=1).unsqueeze(0)
			# 		attention = e_adj / sum_result.transpose(1, 0)
			# 		plt.imshow(feature_img, cmap='gray')
			# 		show_graph_with_labels(attention.cpu().numpy(), slic_arr.numpy(), 35)
			# 		plt.show()
			# 		save_graphfeatruemap(feature, slic_arr, nas_dir, img_name, 'test!')
			# 	# # # show_graph_with_labels(adj_label.cpu().numpy(), slic_arr.numpy(), 87)
			# 	sys.exit()

			# for i, gft in enumerate(gft_down_list):
			# 	save_graphfeatruemap(gft, slic_arr, nas_dir, img_name, 'graph_down_{}'.format(i+1))

			loss = criterion(output.double(), graph_label)
			dsc = save_predict_img(image, label, output, slic_arr, superpix_img, save_dir, img_name)
			
			total_dsc += dsc
			total_loss += loss
			# if cnt == 0:
			print(img_name)
			save_original_img(image, label, superpix_img, nas_dir, img_name)
			for i, feature_map in enumerate(ft_list):
				ft_interpolated = F.interpolate(feature_map, original_size)
				show_featuremaps(ft_interpolated, slic_arr, nas_dir, img_name, 'encoder_{}'.format(i+1))
			for i, gft in enumerate(score_list):
				save_graphfeatruemap(gft, slic_arr, nas_dir, img_name, 'score_{}'.format(i+1))
			for i, gft in enumerate(gft_down_list):
				save_graphfeatruemap(gft, slic_arr, nas_dir, img_name, 'graph_down_{}'.format(i+1))
				# feature_map = reconstruct_graph(torch.mean(gft, dim=1).unsqueeze(1), slic_arr, 0)
				# plt.imshow(feature_map, cmap='gray')
				# plt.imshow(slic_arr.cpu().numpy())
				# show_graph_with_labels(recon_adj_list[i].cpu().numpy(), slic_arr.numpy())

			for i, gft in enumerate(gft_up_list):
				save_graphfeatruemap(gft, slic_arr, nas_dir, img_name, 'graph_up_{}'.format(i+1))
			# if cnt == 3:
			
			# sys.exit()
			cnt += 1

		result_txt = os.path.join(nas_dir, 'result.txt')
		f = open(result_txt, 'w')
		f.write('dsc: {:.4f}'.format(total_dsc))

	print('total dsc : {:.4f}'.format(total_dsc/cnt))
	print('total loss : {:.4f}'.format(total_loss/cnt))

	return total_dsc, total_loss