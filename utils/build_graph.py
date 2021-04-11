import scipy.sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import sys

import torch


def get_node_position(slic_arr, N):
	node_position = {}
	for i in range(N):
		rows, cols = np.where(slic_arr==i)
		y_center = (rows.max() + rows.min()) / 2
		x_center = (cols.max() + cols.min()) / 2
		node_position[i] = (y_center, x_center)

	return node_position

def show_graph_with_labels(adj, slic_arr):
	n, _ = adj.shape

	node_position = {}
	for i in range(n):
		rows, cols = np.where(slic_arr==i)
		y_center = (rows.max() + rows.min()) / 2
		x_center = (cols.max() + cols.min()) / 2
		node_position[i] = (y_center, x_center)
	
	print('n:', n)
	rows, cols = np.where(adj==1)
	G = nx.Graph()
	for i in range(n):
		G.add_node(i, pos=node_position[i])

	for i in range(n):
		# cols = np.where(adj[i] == 1)
		for j in range(i+1, n):
			if adj[i, j] == 1:
				G.add_edge(i, j)
	
	pos = nx.get_node_attributes(G, 'pos')
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edges(G, pos)
	plt.show()

def adj_normalize(adj):
	rowsum = np.array(adj.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	adj = r_mat_inv.dot(adj)
	return adj

def gen_adj(slic_arr):
	slic_arr = slic_arr.data.numpy()
	n_superpixel = np.unique(slic_arr)
	image_size = slic_arr.shape
	adj = np.zeros((len(n_superpixel), len(n_superpixel)), dtype=np.uint8)
	for i in range(image_size[0]):
		_, indexs = np.unique(slic_arr[i], return_index=True)
		near_arr = [slic_arr[i][index] for index in sorted(indexs)]
		
		for j in range(len(near_arr)-1):
			start, end = near_arr[j], near_arr[j+1]
			adj[start, end] = 1.0
			adj[end, start] = 1.0
	for i in range(image_size[1]):
		_, indexs = np.unique(slic_arr[:,i], return_index=True)
		near_arr = [slic_arr[:,i][index] for index in sorted(indexs)]

		for j in range(len(near_arr)-1):
			start, end = near_arr[j], near_arr[j+1]
			adj[start, end] = 1.0
			adj[end, start] = 1.0
	
	# node_position = get_node_position(slic_arr, len(n_superpixel))
	# show_graph_with_labels(adj, node_position)
	# sys.exit()
	
	adj = adj_normalize(adj + sp.eye(adj.shape[0]))
	adj = torch.FloatTensor(adj)
	return adj

def gen_graph(feature_map, slic_arr):
	feature_map = torch.squeeze(feature_map)
	torch_graph = []
	for superpixel_val in torch.unique(slic_arr):
		mask = slic_arr == superpixel_val
		# print(mask.shape)
		# print(feature_map.size())
		# sys.exit()
		temp_map = feature_map[:, mask]
		node = torch.mean(temp_map, dim=1)
		torch_graph.append(node)
	torch_graph = torch.stack(torch_graph)
	
	return torch_graph